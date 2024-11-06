from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EncoderConfig:
    image_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    patch_size: int
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = None
    do_random_mask: bool = True
    mask_ratio: float = 0.75
    head_dim: int = None
    patched_image_height: int = None
    patched_image_width: int = None


    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by the number of attention heads"
        assert self.num_channels == 3, "Number of channels must be 3"
        assert all (value % 2 == 0 for value in [self.image_size, self.hidden_size, self.intermediate_size, self.num_hidden_layers, self.num_attention_heads, self.patch_size]), "All values must be even"
        assert self.attention_dropout >= 0.0 and self.attention_dropout <= 1.0, "Attention dropout must be between 0.0 and 1.0"

        self.num_image_tokens = (self.image_size // self.patch_size) ** 2
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.patched_image_height = self.image_size // self.patch_size
        self.patched_image_width = self.image_size // self.patch_size


class EncoderEmbeddings(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Convolve the image into patches of size `patch_size`
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.position_embedding = nn.Embedding(config.num_image_tokens, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_image_tokens).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class EncoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.num_attention_heads = config.num_attention_heads
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape

        # query, key, value projections
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
       
        q = q.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        k = k.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        v = v.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.out_proj(y)
        return y



class EncoderMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.self_attn = EncoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = EncoderMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = EncoderEmbeddings(config)
        self.encoder = EncoderBlock(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random masking to the input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, embed_dim]
        
        Returns:
            x_masked (torch.Tensor): Masked tensor.
            mask (torch.Tensor): Binary mask showing which tokens were masked (1) or kept (0).
            ids_restore (torch.Tensor): Indices to restore the original order of tokens.
        """
        N, L, D = x.shape  # batch, length, dimension
        len_keep = int(L * (1 - self.config.mask_ratio))  # Number of tokens to keep

        # Generate random noise and shuffle tokens
        noise = torch.rand(N, L)
        ids_shuffle = torch.argsort(noise, dim=1)  # Shuffle by sorting noise
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # Indices to restore original order

        # Keep only a subset of tokens
        ids_keep = ids_shuffle[:, :len_keep]  # Indices of tokens to keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))  # Gather kept tokens

        # Create binary mask (0 for kept, 1 for removed)
        mask = torch.ones([N, L])  # Start with all 1s (all removed)
        mask[:, :len_keep] = 0  # Mark kept tokens as 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # Restore original order of the mask

        # return masked embeddings, binary mask and indices to restore the original order
        return x_masked, mask, ids_restore

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the vision transformer.
        
        Args:
            pixel_values (torch.Tensor): Input image tensor of shape [Batch_Size, Channels, Height, Width].

        Returns:
            torch.Tensor: Final output after encoding and layer normalization.
        """
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        if self.config.do_random_mask:
            # Apply random masking to the embeddings
            masked_hidden_states, mask, ids_restore = self.random_masking(hidden_states)
        else:
            masked_hidden_states = hidden_states
            mask, ids_restore = None, None
            
        # Pass the masked embeddings to the encoder
        last_hidden_state = self.encoder(inputs_embeds=masked_hidden_states)

        # Apply layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Return the output and the binary mask, and the indices to restore the original order
        return last_hidden_state, mask, ids_restore



class EncoderModel(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.vision_model = Encoder(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 