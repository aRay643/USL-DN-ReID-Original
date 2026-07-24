import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import load_clip_to_cpu


class CLIPCLSBaseline(nn.Module):
    """Minimal shared CLIP visual encoder for the two-domain PCL baseline."""

    feature_dim = 768

    def __init__(self, cfg):
        super().__init__()
        height = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        width = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        clip_model = load_clip_to_cpu(
            cfg.MODEL.NAME,
            height,
            width,
            cfg.MODEL.STRIDE_SIZE[0],
        )
        self.image_encoder = clip_model.visual
        # The CLIP visual module computes its native 512-D projection internally,
        # but the baseline never consumes or optimizes that branch.
        self.image_encoder.proj.requires_grad_(False)

    def forward(self, images, modal=None):
        del modal  # Both modalities intentionally share the same visual encoder.
        images = images.to(dtype=self.image_encoder.conv1.weight.dtype)
        _, image_features, _ = self.image_encoder(images)
        cls_feature = image_features[:, 0]
        return F.normalize(cls_feature.float(), p=2, dim=1)

    def load_param(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in checkpoint.items()
        }
        self.load_state_dict(checkpoint, strict=True)


def make_baseline_model(cfg):
    return CLIPCLSBaseline(cfg)
