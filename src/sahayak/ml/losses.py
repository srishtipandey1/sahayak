"""Segmentation losses that ignore the dataset's no-data labels."""

IGNORE_INDEX = 255


def bce_dice_loss(logits, targets, ignore_index: int = IGNORE_INDEX):
    """Compute BCE-with-logits plus soft Dice over labels other than -1/255."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim == targets.ndim + 1 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.shape != targets.shape:
        raise ValueError("Segmentation logits and target masks must have matching shapes.")
    valid = (targets != ignore_index) & (targets != -1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    target_values = targets[valid].to(dtype=logits.dtype)
    if bool(((target_values != 0) & (target_values != 1)).any()):
        raise ValueError("Segmentation targets must contain only 0, 1, or ignored values.")
    valid_logits = logits[valid]
    bce = functional.binary_cross_entropy_with_logits(
        valid_logits,
        target_values,
        reduction="mean",
    )
    probabilities = torch.sigmoid(valid_logits)
    intersection = (probabilities * target_values).sum()
    denominator = probabilities.sum() + target_values.sum()
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return bce + (1.0 - dice)
