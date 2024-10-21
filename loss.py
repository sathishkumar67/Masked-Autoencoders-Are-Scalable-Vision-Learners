import torch.nn as nn

def mae_loss(pred, target, mask):
    """
    Compute the MAE loss for masked patches only.

    Args:
    - pred: Predicted patches, tensor of shape [batch_size, num_patches, channels].
    - target: Ground truth patches, tensor of shape [batch_size, num_patches, channels].
    - mask: Binary mask tensor of shape [batch_size, num_patches], where 1 indicates a masked patch, 0 indicates a visible patch.

    Returns:
    - loss: MSE loss computed only over the masked patches.
    """
    # Reshape mask to be broadcastable over the [batch_size, num_patches, channels] dimension
    mask = mask.unsqueeze(-1)  # Shape: [batch_size, num_patches, 1]

    # Apply the mask: Only keep the masked patches (where mask == 1)
    masked_pred = pred * mask  # Shape: [batch_size, num_patches, channels]
    masked_target = target * mask  # Shape: [batch_size, num_patches, channels]

    # Compute MSE over masked patches
    loss_fn = nn.MSELoss(reduction='sum')  # Sum over all elements

    # Normalize the loss by the number of masked patches
    num_masked_patches = mask.sum()  # Total number of masked patches
    loss = loss_fn(masked_pred, masked_target) / num_masked_patches

    return loss
