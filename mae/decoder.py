from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
import gin

@gin.configurable
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
    norm_eps: float = 1e-8
    attention_dropout: float = 0.0
    do_loss_calculation: bool = True
    use_small_mlp: bool = True
    num_image_tokens: int = None
    head_dim: int = None
    patched_image_height: int = None
    patched_image_width: int = None
    rng_seed: int = 42
    rng_generator: torch.Generator = None

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

        if self.rng_generator is None:
            self.rng_generator = torch.Generator().manual_seed(self.rng_seed)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            dim: The dimension of the input tensor.
            eps: The epsilon value used to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight, self.bias = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim))

    def _norm(self, x) -> torch.Tensor:
        """
        Computes the RMSNorm of a tensor.

        Given an input tensor `x`, compute its RMSNorm by dividing it by the root
        mean square of its elements.

        Args:
            x: The input tensor.

        Returns:
            The RMSNorm of the input tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:        
        """
        Computes the RMSNorm of a tensor and applies a learnable scale factor.

        Args:
            x: The input tensor.

        Returns:
            The RMSNorm of the input tensor multiplied by a learnable scale factor.
        """
        return self._norm(x.float()).type_as(x) * self.weight + self.bias

class DecoderAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        """
        Initializes the DecoderAttention class.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.

        The initializer sets up key, query, value projections for all heads in a batch and an output projection.
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
        Forward pass of the DecoderAttention module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after the scaled dot product attention and the output linear layer.
        """
        B, T, C = hidden_states.shape

        # query, key, value projections
        q, k, v = self.qkv_proj(hidden_states).split(self.config.hidden_size, dim=-1)

        # reshape q, k, v for multi-head attention
        q = q.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        k = k.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        v = v.view(B, T, self.config.num_attention_heads, C // self.config.num_attention_heads).transpose(1, 2) 
        
        # attention and out projection
        # [Batch_Size, Num_Patches, Dim]
        return self.out_proj(F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.config.attention_dropout).transpose(1, 2).contiguous().view(B, T, C))


class DecoderMLPLight(nn.Module): # This is lightweight MLP if needed use gpt2_turbo MLP
    def __init__(self, config: DecoderConfig) -> None:
        """
        __init__ method of the DecoderMLP class.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.

        Initializes the DecoderMLP class with two linear layers. The first layer projects the input to the intermediate size, and the second layer projects the output to the hidden size.
        """
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.fc2.SCALE_INIT = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DecoderMLP module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after the two linear layers with GELU activation.
        """
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class DecoderMLPLarge(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        """
        __init__ method of the DecoderMLPLarge class.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.

        Initializes the DecoderMLPLarge class with three linear layers. The first layer projects the input to the intermediate size, the second layer projects the intermediate size to the hidden size, and the third layer projects the hidden size to the intermediate size.
        """
        super().__init__()
        self.config = config
        
        # projections
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.down_proj.SCALE_INIT = 1
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DecoderMLPLarge module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after the three linear layers with SiLU activation.
        """
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        """
        Initializes the DecoderLayer class with self-attention, normmalization, and MLP components.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.

        The initializer sets up a self-attention module, two normmalization modules, and an MLP module.
        """
        super().__init__()
        self.config = config

        self.self_attn = DecoderAttention(config)
        self.norm_1 = RMSNorm(self.config.hidden_size, eps=config.norm_eps)
        if config.use_small_mlp:
            self.mlp = DecoderMLPLight(config)
        else:
            self.mlp = DecoderMLPLarge(config)
        self.norm_2 = RMSNorm(self.config.hidden_size, eps=config.norm_eps)

    # Ignore copy
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DecoderLayer module.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after self-attention and mlp block.
        """
        # x: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        x = x + self.self_attn(self.norm_1(x))
        # x: [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        x = x + self.mlp(self.norm_2(x))
        # x: [Batch_Size, Num_Patches, Embed_Dim]
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        """
        Initializes the DecoderBlock class with a list of DecoderLayer objects.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.

        The initializer sets up a list of DecoderLayer objects, each of which contains a self-attention module, two normmalization modules, and an MLP module.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DecoderBlock module.

        Args:
            inputs_embeds (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: Final output after the decoder block.
        """
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = decoder_layer(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        """
        Initializes the Decoder class with a projection layer, mask token, position embedding, decoder block, post normalization, and prediction layer.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.

        The initializer sets up a projection layer, mask token, position embedding, decoder block, post normalization, and prediction layer.
        """
        super().__init__()
        self.config = config

        # check if the input projection dimension is equal to the embedding dimension
        # if not, add a linear layer to project the input to the embedding dimension
        # else, use the identity layer
        if self.config.in_proj_dim != self.config.hidden_size:
            self.projector = nn.Linear(self.config.in_proj_dim, self.config.hidden_size, bias=True)
            self.projector_norm = RMSNorm(self.config.hidden_size, eps=config.norm_eps)
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
        self.post_norm = RMSNorm(self.config.hidden_size, eps=config.norm_eps)

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

        if ids_restore is None:
            # return the encoded tokens if there is no need to restore the original order
            return encoded_tokens, mask, ids_restore 
        else:
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
        x, mask, _ = self.reconstruct_sequence(x)

        # pass the output through the subsequent layers
        x = self.predictor(self.post_norm(self.decoder(x)))

        # calculate the loss
        if self.config.do_loss_calculation:
            loss = self.loss(target=target, prediction=x, mask=mask)
            return self.unpatchify(x), loss 
        else:
            return self.unpatchify(x), None


class DecoderModel(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        """
        Initializes the DecoderModel class.

        Args:
            config (DecoderConfig): Configuration object containing model hyperparameters.
        
        The initializer sets up a vision model using the given configuration.
        """
        super().__init__()
        self.config = config
        self.vision_model = Decoder(config)

        # Initialize weights 
        self.apply(self.__init__weights)
        
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.num_hidden_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.config.rng_generator.manual_seed(self.config.rng_seed))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.config.rng_generator.manual_seed(self.config.rng_seed))
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.config.rng_generator.manual_seed(self.config.rng_seed))
            nn.init.zeros_(module.bias)
            
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor) -> Tuple:
        """
        Forward pass of the DecoderModel.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing the encoded representation, the binary mask, and the indices to restore the original order.
            target (torch.Tensor): Target tensor of shape [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            Tuple: Tuple containing the prediction tensor of shape [Batch_Size, Num_Patches, Embed_Dim] and the loss tensor.
        """
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(x, target) 