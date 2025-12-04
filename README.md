# DETR Forward Hook — Option 1B

This project implements Option 1B of the DETR moved-object detection task:
using a forward hook to extract intermediate CNN features from DETR's backbone,
computing a feature difference (B - A), and sending this difference into the
transformer module only.

## Pipeline
1. Extract features for each image using DETR’s ResNet backbone via forward hook.
2. Compute feature difference.
3. Flatten and feed difference into DETR transformer + heads (not full DETR).
4. Train classifier + box regression heads.

## Run
sbatch run_main.sbatch

shell
Copy code

## Directory Structure
DETR-Forward-Hook-/
├── config.py
├── dataset.py
├── model.py
├── trainer.py
├── evaluator.py
├── main.py
├── run_main.sbatch
├── requirements.txt
├── README.md
├── data/
│ ├── base/cv_data_hw2/
│ └── matched_annotations/
├── checkpoints/
└── outputs/