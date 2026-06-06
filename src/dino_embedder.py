"""DINOv3 appearance embedder for bounding-box crops (ReID).

Produces one L2-normalised global descriptor per crop, used to keep a stable
identity on the tracked person across MOG2 dropouts / ByteTrack ID switches.

Loading mirrors the proven pattern in the dinov3 worktree's DenseMatcher
(facebook/dinov3-vits16-pretrain-lvd1689m, GPU fp16, imagenet norm, patch
tokens), but for ReID we MEAN-POOL the patch tokens into a single vector rather
than matching them densely.

The gated model needs a Hugging Face token: set `HF_TOKEN` in the environment
(read here, never hardcoded). The target is only a few pixels, so each bbox is
context-padded before embedding to give the backbone something to describe.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from transformers import AutoModel

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DinoEmbedder:
    def __init__(
        self,
        repo: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        longside: int = 112,        # 16 * 7, a patch multiple
        crop_min_px: int = 64,      # context-pad tiny boxes up to this
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        token = os.environ.get("HF_TOKEN")  # gated repo; never hardcode
        self.model = AutoModel.from_pretrained(
            repo, dtype=self.dtype, token=token
        ).to(self.device).eval()
        self.patch = getattr(self.model.config, "patch_size", 16)
        self.longside = max(self.patch, round(longside / self.patch) * self.patch)
        self.crop_min_px = crop_min_px
        self._mean = torch.tensor(_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.dim = int(self.model.config.hidden_size)

    def context_crop(self, frame_bgr: np.ndarray, box_xyxy) -> np.ndarray:
        """Crop the box expanded to >= crop_min_px on each side (clamped)."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max((x2 - x1), (y2 - y1), self.crop_min_px) / 2.0
        ax1 = int(max(0, cx - half)); ay1 = int(max(0, cy - half))
        ax2 = int(min(w, cx + half)); ay2 = int(min(h, cy + half))
        crop = frame_bgr[ay1:ay2, ax1:ax2]
        if crop.size == 0:
            crop = frame_bgr[max(0, int(cy)):int(cy) + 1, max(0, int(cx)):int(cx) + 1]
        return crop

    def _to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        s = self.longside  # square input keeps it simple for small crops
        rgb = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (s, s),
                         interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return x

    @torch.inference_mode()
    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR crops -> (N, D) L2-normalised float32."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        batch = torch.stack([self._to_tensor(c) for c in crops]).to(self.device, self.dtype)
        batch = (batch - self._mean) / self._std
        gh = gw = self.longside // self.patch
        lhs = self.model(pixel_values=batch).last_hidden_state   # (N, T, D)
        patch_tok = lhs[:, -gh * gw:, :]                         # drop CLS + registers
        desc = patch_tok.float().mean(dim=1)                     # mean-pool -> (N, D)
        desc = torch.nn.functional.normalize(desc, dim=1)
        return desc.cpu().numpy().astype(np.float32)

    def embed_boxes(self, frame_bgr: np.ndarray, boxes_xyxy) -> np.ndarray:
        crops = [self.context_crop(frame_bgr, b) for b in boxes_xyxy]
        return self.embed_crops(crops)


if __name__ == "__main__":
    # smoke test: load + embed a couple of dummy crops
    import time
    emb = DinoEmbedder()
    print(f"loaded {emb.model.config.model_type} | dim={emb.dim} | patch={emb.patch} "
          f"| longside={emb.longside} | device={emb.device}")
    dummy = [np.random.randint(0, 255, (12, 8, 3), np.uint8),
             np.random.randint(0, 255, (40, 30, 3), np.uint8)]
    t0 = time.perf_counter()
    v = emb.embed_crops(dummy)
    dt = (time.perf_counter() - t0) * 1000
    print(f"embeddings shape={v.shape} norms={np.linalg.norm(v, axis=1)} "
          f"finite={np.isfinite(v).all()} | {dt:.1f} ms for {len(dummy)} crops")
