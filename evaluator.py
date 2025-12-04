# evaluator.py

import torch
from trainer import box_cxcywh_to_xyxy, iou_xyxy

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader, iou_thresh=0.5):
        self.model.eval()

        TP = FP = FN = 0

        for A, B, targets in loader:
            A, B = A.to(self.device), B.to(self.device)
            pred = self.model(A, B)

            logits = pred["pred_logits"]
            boxes = pred["pred_boxes"]

            for b in range(len(targets)):
                gt_boxes = targets[b]["boxes"]
                gt_labels = targets[b]["labels"]

                pb = boxes[b]
                pl = logits[b]

                probs = pl.softmax(-1)
                scores, labels = probs.max(-1)

                pb_xy = box_cxcywh_to_xyxy(pb)
                gt_xy = box_cxcywh_to_xyxy(gt_boxes)

                ious = iou_xyxy(gt_xy, pb_xy)

                for i in range(gt_boxes.size(0)):
                    best = ious[i].max().item()
                    if best >= iou_thresh:
                        TP += 1
                    else:
                        FN += 1

                FP += max(0, pb.size(0) - gt_boxes.size(0))

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        return precision, recall
