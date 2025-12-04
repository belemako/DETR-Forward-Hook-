# config.py

class Config:
    # Paths
    DATA_DIR = "data/base/cv_data_hw2"
    MATCHED_ANN_DIR = "data/matched_annotations"
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "outputs"

    # Model / dataset
    DETR_NAME = "facebook/detr-resnet-50"
    NUM_CLASSES = 6  # Unknown=0, person=1, car=2, vehicle=3, object=4, bike=5

    # Training
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LR = 1e-5
    WEIGHT_DECAY = 1e-4

    IMAGE_SIZE = (480, 640)

    DEVICE = "cuda"
    PRINT_FREQ = 20
