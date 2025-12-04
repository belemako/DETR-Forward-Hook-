# dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VIRATMovedObjectsDataset(Dataset):
    """
    Pair-based dataset for moved-object detection.
    Expects files:
        <id>_A.jpg
        <id>_B.jpg
        matched_annotations/<id>.txt
    """

    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform

        self.ids = sorted(
            f[:-4] for f in os.listdir(ann_dir) if f.endswith(".txt")
        )

    def __len__(self):
        return len(self.ids)

    def _load_images(self, base_id):
        imgA = Image.open(os.path.join(self.img_dir, base_id + "_A.jpg")).convert("RGB")
        imgB = Image.open(os.path.join(self.img_dir, base_id + "_B.jpg")).convert("RGB")
        return imgA, imgB

    def _load_targets(self, base_id, img_w, img_h):
        boxes = []
        labels = []

        path = os.path.join(self.ann_dir, base_id + ".txt")
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i in range(0, len(lines), 2):
            _, x, y, w, h, c = lines[i + 1].split()
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            c = int(c)

            cx = (x + x + w) / 2 / img_w
            cy = (y + y + h) / 2 / img_h
            bw = w / img_w
            bh = h / img_h

            boxes.append([cx, cy, bw, bh])
            labels.append(c)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": base_id
        }

    def __getitem__(self, idx):
        base_id = self.ids[idx]
        imgA, imgB = self._load_images(base_id)
        w, h = imgA.size

        target = self._load_targets(base_id, w, h)

        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return imgA, imgB, target


def collate_fn(batch):
    imgA, imgB, targets = zip(*batch)
    return torch.stack(imgA), torch.stack(imgB), list(targets)
