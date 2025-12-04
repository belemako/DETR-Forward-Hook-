# model.py

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection


class _DetrTransformerProxy(nn.Module):
    """
    Compatibility wrapper for transformers versions where `DetrModel` no longer
    exposes `.transformer`. It mirrors the old `(hs, memory)` output tuple.
    """

    def __init__(self, detr_model):
        super().__init__()
        if not (hasattr(detr_model, "encoder") and hasattr(detr_model, "decoder")):
            raise AttributeError(
                "DetrModel lacks both `.transformer` and encoder/decoder blocks."
            )
        self.detr_model = detr_model

    def forward(self, src, query_embed):
        if query_embed is None:
            raise ValueError("`query_embed` must be provided to the transformer proxy.")

        tgt = torch.zeros_like(query_embed)
        memory = self.detr_model.encoder(src, src_key_padding_mask=None, pos=None)
        hs = self.detr_model.decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        return hs, memory


class DETRFeatureDiffOption1B(nn.Module):
    """
    Option 1B: Use DETR's internal backbone via forward hook,
    compute feature difference, send to transformer only.
    """

    def __init__(self, detr_name, num_classes):
        super().__init__()

        self.base = DetrForObjectDetection.from_pretrained(detr_name)
        self.backbone = self.base.model.backbone

        transformer = getattr(self.base.model, "transformer", None)
        if transformer is None:
            transformer = _DetrTransformerProxy(self.base.model)
        self.transformer = transformer

        self.query_embed = self.base.model.query_position.weight  # (num_queries, d_model)

        d_model = self.base.config.d_model
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = self.base.bbox_predictor

        self._buffer = []

        hook_module = getattr(self.backbone, "conv_encoder", self.backbone)
        hook_module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._buffer.append(out)

    def _extract_features(self, img):
        self._buffer = []
        _ = self.backbone(img)

        feat = self._buffer[-1]
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if hasattr(feat, "tensors"):
            feat = feat.tensors
        return feat

    def forward(self, imgA, imgB):
        featA = self._extract_features(imgA)
        featB = self._extract_features(imgB)
        diff = featB - featA  # [B, C, H, W]

        B, C, H, W = diff.shape

        src = diff.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.unsqueeze(1).repeat(1, B, 1)

        hs = self.transformer(src, query_embed=query_embed)[0]  # decoder output
        hs = hs[-1]  # [B, num_queries, C]

        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
