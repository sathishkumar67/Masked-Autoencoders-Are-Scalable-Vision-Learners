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
    layer_norm_eps: float = 1e-8
    attention_dropout: float = 0.0
    do_random_mask: bool = True
    mask_ratio: float = 0.75
    head_dim: int = None
    patched_image_height: int = None
    patched_image_width: int = None
    num_image_tokens: int = None

    def __post_init__(self):
        # Check if the values are valid
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by the number of attention heads"
        assert self.num_channels == 3, "Number of channels must be 3"
        assert all (value % 2 == 0 for value in [self.image_size, self.hidden_size, self.intermediate_size, self.num_hidden_layers, self.num_attention_heads, self.patch_size]), "All values must be even"
        assert self.attention_dropout >= 0.0 and self.attention_dropout <= 1.0, "Attention dropout must be between 0.0 and 1.0"

        # Calculate values after initialization
        self.num_image_tokens = (self.image_size // self.patch_size) ** 2
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.patched_image_height = self.image_size // self.patch_size
        self.patched_image_width = self.image_size // self.patch_size


class EncoderEmbeddings(nn.Module):
    def __init__(self, config: EncoderConfig):
        """
        __init__ method of the EncoderEmbeddings class.

        Args:
            config (EncoderConfig): Configuration object containing model hyperparameters.

        Initializes the EncoderEmbeddings class with a convolutional layer to project the input image into patches of size `patch_size`
        and a positional embedding layer to add positional embeddings to the patches.
        """
        super().__init__()
        self.config = config

        # Convolve the image into patches of size `patch_size`
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid", # This indicates no padding is added
            bias=True # adding bias
        )

        # Positional embeddings for the patches
        self.position_embedding = nn.Embedding(config.num_image_tokens, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_image_tokens).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass of the EncoderEmbeddings module.

        Args:
            pixel_values (torch.FloatTensor): Input image tensor of shape [Batch_Size, Channels, Height, Width].

        Returns:
            torch.Tensor: Output tensor of shape [Batch_Size, Num_Patches, Embed_Dim] after applying convolution-based patch embedding 
            and adding positional embeddings.
        """
        # [Batch_Size, Num_Patches, Embed_Dim]
        return self.patch_embedding(pixel_values).flatten(2).transpose(1, 2) + self.position_embedding(self.position_ids)


class EncoderAttention(nn.Module):
    def __init__(self, config):
        """
        __init__ method of the EncoderAttention class.

        Args:
            config (EncoderConfig): Configuration object containing model hyperparameters.

        Initializes the EncoderAttention class with key, query, value projections and output projection.
        """
        super().__init__()
        self.config = config

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        # output projection 
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj.SCALE_INIT = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderAttention module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after the scaled dot product attention and the output linear layer.
        """
        B, T, C = hidden_states.shape

        # query, key, value projections
        q, k, v = self.qkv_proj(hidden_states).split(self.config.hidden_size, dim=-1)
       
       # reshape for multihead self attention
        q = q.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        k = k.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        v = v.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        
        # attention and out projection
        # [Batch_Size, Num_Patches, Dim]
        return self.out_proj(F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.config.attention_dropout).transpose(1, 2).contiguous().view(B, T, C))



class EncoderMLP(nn.Module): # This is lightweight MLP if needed use gpt2_turbo MLP
    def __init__(self, config):
        """
        __init__ method of the EncoderMLP class.

        Args:
            config (EncoderConfig): Configuration object containing model hyperparameters.

        Initializes the EncoderMLP class with two linear layers. The first layer projects the input to the intermediate size, and the second layer projects the output to the hidden size.
        """
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.fc2.SCALE_INIT = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:        
        """
        Forward pass of the EncoderMLP module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after the two linear layers with GELU activation.
        """
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))



class EncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        """
        Initialize an EncoderLayer instance.

        Args:
            config (EncoderConfig): Configuration object containing model hyperparameters.

        Initializes the EncoderLayer with self-attention, layer normalization, and MLP components.
        """
        super().__init__()
        self.config = config
        self.self_attn = EncoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = EncoderMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderLayer module.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after self-attention and mlp block.
        """
        # x: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        x = x + self.self_attn(self.layer_norm1(x))
        # x: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        x = x + self.mlp(self.layer_norm2(x))
        # x: [Batch_Size, Num_Patches, Embed_Dim]
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        """
        Constructor for the EncoderBlock class.

        Args:
            config (EncoderConfig): Configuration object containing model hyperparameters.

        Initializes the EncoderBlock class with a list of EncoderLayer objects.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            inputs_embeds (torch.Tensor): input embeddings of shape (Batch_Size, Num_Patches, Embed_Dim)

        Returns:
            torch.Tensor: output embeddings of shape (Batch_Size, Num_Patches, Embed_Dim)
        """
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        """
        __init__ method of the Encoder class.

        Args:
            config (EncoderConfig): Configuration for the Encoder.

        Initializes the Encoder class with the given configuration. The Encoder consists of an embedding layer, a block of encoder layers, and a layer normalization layer.
        """
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
        ids_shuffle = torch.argsort(torch.rand(N, L), dim=1)  # Shuffle by sorting noise
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

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, None, None]:
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
            # No masking applied to the embeddings
            masked_hidden_states = hidden_states
            mask, ids_restore = None, None

        # Return the output and the binary mask, and the indices to restore the original order
        return self.post_layernorm(self.encoder(masked_hidden_states)), mask, ids_restore


class EncoderModel(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        """
        Constructor for the EncoderModel.

        Args:
            config (EncoderConfig): Configuration object containing model hyperparameters.

        Returns:
            contextual_embeddings (torch.Tensor): Output tensor of the model.
            mask (torch.Tensor): Binary mask showing which tokens were masked (1) or kept (0).
            ids_restore (torch.Tensor): Indices to restore the original order of tokens.
        """
        super().__init__()
        self.config = config
        self.vision_model = Encoder(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, None, None]:
        """
        Forward pass of the vision transformer model.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape [Batch_Size, Channels, Height, Width].

        Returns:
            contextual_embeddings (torch.Tensor): Output tensor of the model.
            mask (torch.Tensor): Binary mask showing which tokens were masked (1) or kept (0).
            ids_restore (torch.Tensor): Indices to restore the original order of tokens.
        """
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 