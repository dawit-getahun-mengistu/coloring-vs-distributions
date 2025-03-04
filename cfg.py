import torch

DATASET_PATH = 'data/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
IMG_SIZE = 32


fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
