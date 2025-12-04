# main.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from config import Config
from dataset import VIRATMovedObjectsDataset, collate_fn
from model import DETRFeatureDiffOption1B
from trainer import Trainer
from evaluator import Evaluator

def main():
    device = torch.device(
        Config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    dataset = VIRATMovedObjectsDataset(
        img_dir=Config.DATA_DIR,
        ann_dir=Config.MATCHED_ANN_DIR,
        transform=transform
    )

    n = len(dataset)
    train_n = int(0.8 * n)
    test_n = n - train_n

    train_set, test_set = random_split(dataset, [train_n, test_n])

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    model = DETRFeatureDiffOption1B(Config.DETR_NAME, Config.NUM_CLASSES)
    trainer = Trainer(model, train_loader, device, Config.LR, Config.WEIGHT_DECAY)

    trainer.train(Config.NUM_EPOCHS)

    torch.save(model.state_dict(), f"{Config.CHECKPOINT_DIR}/detr_option1b.pth")

    evaluator = Evaluator(model, device)
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()
