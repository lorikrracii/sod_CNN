import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')

#raw data
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw", "images")
RAW_MASK_DIR = os.path.join(DATA_DIR, "raw", "masks")

#i splitim data directories
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")

VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")

TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

#ku i bojim save models dhe results
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


#TRAINING HYPERPARAMETERS
IMAGE_SIZE = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

#Model config
ENCODER_CHANNELS = [32 , 64 , 128]
DECODER_CHANNELS = [128 , 64, 32]

DEVICE = "cuda"
SEED = 42