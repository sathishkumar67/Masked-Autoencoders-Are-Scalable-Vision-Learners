from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DecoderConfig:
    image_size: int
    in_proj_dim: int # Input projection dimension
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    patch_size: int
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = None


class DecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape

        # query, key, value projections
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
       
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) 
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) 
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) 
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.out_proj(y)
        return y


class DecoderMLP(nn.Module):
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


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = DecoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = DecoderMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

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
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
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


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = decoder_layer(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.hidden_size
        in_proj_dim = config.in_proj_dim
        out_channels = config.num_channels

        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.height = self.image_size // self.patch_size
        self.width = self.image_size // self.patch_size

        # check if the input projection dimension is equal to the embedding dimension
        # if not, add a linear layer to project the input to the embedding dimension
        # else, use the identity layer
        if in_proj_dim != embed_dim:
            self.projector = nn.Linear(in_proj_dim, embed_dim, bias=True)
            self.projector_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.projector = nn.Identity()
            self.projector_norm = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(self.num_positions, embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

        self.decoder = DecoderBlock(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.reverse_patch_embedding = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=self.config.num_channels,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size)
    

    def reconstruct_sequence(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct the original sequence from the masked sequence.
        
        Args:
            x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the output tensor, the binary mask, and the indices to restore the original order.

        Returns:
            torch.Tensor: Reconstructed sequence.
        """
        # Unpack the tuple
        encoded_tokens, mask, ids_restore = x

        # project the encoded tokens
        encoded_tokens = self.projector(encoded_tokens)
        # normalize the encoded tokens
        encoded_tokens = self.projector_norm(encoded_tokens)

        # append the mask token to the encoded tokens
        num_mask_tokens = ids_restore.shape[1] - encoded_tokens.shape[1] # calculate the number of mask tokens to be needed
        mask_tokens = self.mask_token.repeat(encoded_tokens.shape[0], num_mask_tokens, 1) # repeat the mask token for the batch
        encoded_tokens_masked = torch.cat([encoded_tokens, mask_tokens], dim=1) # concatenate the mask tokens to the encoded tokens

        # unshuflle the tokens to the original order
        encoded_tokens_masked = torch.gather(encoded_tokens_masked, 1, index=ids_restore.unsqueeze(-1).repeat(1, 1, encoded_tokens.shape[2]))

        return encoded_tokens_masked


    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Forward pass of the Vision Transformer.
        Args:
        x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the encoded representation, the binary mask, and the indices to restore the original order.

        Returns:
            torch.Tensor: decoded sequence.
        """

        # Reconstruct the original sequence
        x = self.reconstruct_sequence(x)

        # pass the reconstructed sequence through the decoder
        x = self.decoder(x)

        # apply layer normalization
        x = self.post_layernorm(x)

        # reshape the tensor
        x = x.transpose(1, 2).contiguous().view(-1, self.hidden_size, self.height, self.width)

        # reverse the patch embedding
        x = self.reverse_patch_embedding(x)

        return x


class DecoderModel(nn.Module):

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.vision_model = Decoder(config)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(x) 