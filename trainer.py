# trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import Config

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([
        cx - w * 0.5,
        cy - h * 0.5,
        cx + w * 0.5,
        cy + h * 0.5
    ], dim=-1)

def iou_xyxy(b1, b2):
    """
    b1: [N,4], b2: [M,4]
    returns IoU matrix [N,M]
    """
    N = b1.size(0)
    M = b2.size(0)
    if N == 0 or M == 0:
        return torch.zeros(N, M, device=b1.device)

    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    return inter / (area1[:, None] + area2[None, :] - inter + 1e-6)

class Trainer:
    def __init__(self, model, loader, device, lr, wd, evaluator=None, eval_loader=None):
        self.model = model.to(device)
        self.loader = loader
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.evaluator = evaluator
        self.eval_loader = eval_loader

    def _loss(self, pred, targets):
        logits = pred["pred_logits"]
        boxes = pred["pred_boxes"]

        total = 0.0

        for b in range(logits.size(0)):
            tgt = targets[b]
            tb = tgt["boxes"].to(self.device)
            tl = tgt["labels"].to(self.device)

            if tb.size(0) == 0:
                continue

            pl = logits[b]
            pb = boxes[b]

            pb_xy = box_cxcywh_to_xyxy(pb)
            tb_xy = box_cxcywh_to_xyxy(tb)

            ious = iou_xyxy(tb_xy, pb_xy)
            idx = ious.argmax(dim=1)

            cls_loss = F.cross_entropy(pl[idx], tl)
            box_loss = F.l1_loss(pb[idx], tb)

            total += cls_loss + box_loss

        return total

    def train(self, epochs):
        self.model.train()

        for ep in range(epochs):
            total = 0.0
            for i, (A, B, tgt) in enumerate(tqdm(self.loader, desc=f"Epoch {ep+1}/{epochs}")):
                A, B = A.to(self.device), B.to(self.device)

                pred = self.model(A, B)
                loss = self._loss(pred, tgt)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total += loss.item()

                if (i + 1) % Config.PRINT_FREQ == 0:
                    print(f"[Iter {i+1}] Loss: {loss.item():.4f}")

            avg_loss = total / len(self.loader)
            print(f"Epoch {ep+1} Avg Loss: {avg_loss:.4f}")

            if self.evaluator is not None and self.eval_loader is not None:
                precision, recall = self.evaluator.evaluate(self.eval_loader)
                print(f"[Epoch {ep+1}] Eval Precision: {precision:.4f}, Recall: {recall:.4f}")
                self.model.train()
