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
    head_dim: int = None
    do_loss_calculation: bool = True
    do_norm_pix_loss: bool = False
    patched_image_height: int = None
    patched_image_width: int = None

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by the number of attention heads"
        assert self.num_channels == 3, "Number of channels must be 3"
        assert all (value % 2 == 0 for value in [self.image_size, self.in_proj_dim, self.hidden_size, self.intermediate_size, self.num_hidden_layers, self.num_attention_heads, self.patch_size]), "All values must be even"
        assert self.attention_dropout >= 0.0 and self.attention_dropout <= 1.0, "Attention dropout must be between 0.0 and 1.0"

        self.num_image_tokens = (self.image_size // self.patch_size) ** 2
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.patched_image_height = self.image_size // self.patch_size
        self.patched_image_width = self.image_size // self.patch_size


class DecoderAttention(nn.Module):
    """Multi-Head Attention module."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.k_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.out_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape

        # query, key, value projections
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
       
        q = q.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        k = k.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        v = v.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.config.attention_dropout) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.out_proj(y)
        return y


class DecoderMLP(nn.Module):
    """MLP module."""
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
    """Decoder layer module."""
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.self_attn = DecoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DecoderMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)

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


class DecoderBlock(nn.Module):
    """Decoder block module."""
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = decoder_layer(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # check if the input projection dimension is equal to the embedding dimension
        # if not, add a linear layer to project the input to the embedding dimension
        # else, use the identity layer
        if self.config.in_proj_dim != self.config.hidden_size:
            self.projector = nn.Linear(self.config.in_proj_dim, self.config.hidden_size, bias=True)
            self.projector_norm = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.projector = nn.Identity()
            self.projector_norm = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.position_embedding = nn.Embedding(self.config.num_image_tokens, self.config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.config.num_image_tokens).expand((1, -1)),
            persistent=False,
        )

        self.decoder = DecoderBlock(config)
        self.post_layernorm = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)

        # linear layer to project the output to the number of channels
        self.predictor = nn.Linear(self.config.hidden_size, self.config.patch_size ** 2 * self.config.num_channels, bias=True)
    

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

        return encoded_tokens_masked, mask, ids_restore
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch_Size, Channels, Height, Width]
        output: [Batch_Size, Num_Patches, Patch_Size ** 2 * Channels]
        """

        # reshape the tensor
        x = x.reshape(-1, self.config.num_channels, self.config.patched_image_height, self.config.patch_size, self.config.patched_image_width, self.config.patch_size)

        # perform einsum operation
        x = torch.einsum('nchpwq->nhwpqc', x)

        # reshape the tensor
        x = x.reshape(-1, self.config.patched_image_height * self.config.patched_image_width, self.config.patch_size **2 * self.config.num_channels)

        # (Batch_Size, Num_Patches, Patch_Size ** 2 * Channels)
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch_Size, Num_Patches, Patch_Size ** 2 * Channels]
        output: [Batch_Size, Channels, Height, Width]
        """

        # reshape the tensor
        x = x.reshape(-1, self.config.patched_image_height, self.config.patched_image_width, self.config.patch_size, self.config.patch_size, self.config.num_channels)

        # perform einsum operation
        x = torch.einsum('nhwpqc->nchpwq', x)

        # reshape the tensor
        x = x.reshape(-1, self.config.num_channels, self.config.patched_image_height * self.config.patch_size, self.config.patched_image_width * self.config.patch_size)

        # (Batch_Size, Channels, Height, Width)
        return x
    
    def loss(self, target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss of the decoder model.
        Args:
            target (torch.Tensor): Target tensor of shape [Batch_Size, Channels, Height, Width].
            prediction (torch.Tensor): Prediction tensor of shape [Batch_Size, Num_Patches, Patch_Size ** 2 * Channels].
            mask (torch.Tensor): Binary mask of shape [Batch_Size, Num_Patches]. 0 is keep, 1 is remove

        Returns:
            torch.Tensor: Loss tensor of shape [].
        """
        # calculate the loss
        target = self.patchify(target)

        
        # do normalization if needed
        # if self.config.do_norm_pix_loss:                                          # need to check this
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1e-6) ** 0.5
        #     prediction = (prediction - mean) / (var + 1e-6) ** 0.5

        # manual loss calculation
        # loss = (prediction - target) ** 2
        # loss = loss.mean(dim=-1)  # mean over all channels
        # loss = (loss * mask).sum() / mask.sum()  # mean only over non-ignored pixels

        # hybrid approach
        # Step 1: Calculate the element-wise MSE loss without reduction
        # loss = F.mse_loss(prediction, target, reduction='none')  # Shape: [Batch_Size, Num_Patches, Patch_Size ** 2 * Channels]
        # # Step 2: Take the mean across the channel dimension to match mask shape
        # loss = loss.mean(dim=-1)  # Shape: [Batch_Size, Num_Patches]
        # # Step 3: Apply the mask and compute the mean loss over the masked tokens
        # masked_loss = (loss * mask).sum() / mask.sum()

        # direct approach
        # Expand mask to match the shape of the patches
        mask_expanded = mask.unsqueeze(-1).expand_as(prediction)

        # Filter out only the masked patches (where mask is 1)
        masked_prediction = prediction[mask_expanded == 1].view(-1, prediction.shape[-1])
        masked_target = target[mask_expanded == 1].view(-1, target.shape[-1])

        # Calculate mean squared error only on the masked patches
        loss = F.mse_loss(masked_prediction, masked_target, reduction='mean')

        return loss


    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        Args:
        x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the encoded representation, the binary mask, and the indices to restore the original order.

        Returns:
            torch.Tensor: decoded sequence.
        """

        # Reconstruct the original sequence
        x, mask, ids_restore = self.reconstruct_sequence(x)

        # pass the reconstructed sequence through the decoder
        x = self.decoder(x)

        # apply layer normalization
        x = self.post_layernorm(x)

        # pass the output through the predictor
        x = self.predictor(x)

        # calculate the loss
        if self.config.do_loss_calculation:
            loss = self.loss(target=target, prediction=x, mask=mask)
            return self.unpatchify(x), loss
        else:
            return self.unpatchify(x)


class DecoderModel(nn.Module):

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.vision_model = Decoder(config)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(x, target) 