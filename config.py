classes_name =  [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

ANCHORS = [
    [0.57273, 0.677385],
    [1.87446, 2.06253],
    [3.33843, 5.47434],
    [7.88282, 3.52778],
    [9.77052, 9.16828]
]

CLASS_WEIGHT = [
    3.02114804, 2.39808153, 1.66944908, 2.51256281, 1.58982512, 3.67647059,
    0.61124694, 2.57069409, 0.70175439, 2.80898876, 3.23624595, 1.86567164,
    2.48138958, 2.56410256, 0.18545994, 1.61290323, 2.83286119, 2.38663484,
    3.04878049, 2.73224044,
]

# YOLO version
VERSION = 1

# backbone
BACKBONE = 'vgg'

# VGG
VGG_MEANS = [103.939, 116.779, 123.68] # BGR

# dataset
DATA_PATH = './data/pascal_voc_training_data.txt'
DATA_PATH_TEST = './data/pascal_voc_testing_data.txt'
IMAGE_DIR = # Directory you place your training image data
IMAGE_DIR_TEST = # Directory you place your testing image data

# common params
IMAGE_SIZE = 448 if VERSION == 1 else 416
NUM_CLASSES = 20
MAX_OBJECTS_PER_IMAGE = 20

# YOLO params
CELL_SIZE = 7 if VERSION == 1 else 13
BOXES_PER_CELL = 2 if VERSION == 1 else 5
OBJECT_SCALE = 1
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1
COORD_SCALE = 5

# training params
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LEARNING_RATE_MIN = 1e-5
LEARNING_RATE_WARMUP = 1e-4
EPOCHS = 600
EPOCHS_WARMUP = 16
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LINEAR_SCALE = 64 / BATCH_SIZE
WARMUP = True

# testing params
PROGRESSBAR_WIDTH = 40

# checkpoint
CKPT_DIR = './checkpoints'
