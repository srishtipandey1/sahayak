"""SMP U-Net model factory with a ResNet-34 encoder."""


def build_model(encoder_weights: str | None = "imagenet"):
    """Create the requested one-logit U-Net without importing SMP at package load."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError(
            'Install the optional machine-learning dependencies with `pip install -e ".[ml]"`.'
        ) from error
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
