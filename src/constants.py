"""
Constants and magic numbers used throughout the codebase.
Centralizing these values makes the code more maintainable and self-documenting.
"""

# Token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Token strings
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<UNK>'

# Color token indices (used for metrics collection)
COLOR_TOKENS = [4, 6, 9, 10, 11, 14]

# Shape token indices (used for metrics collection)
SHAPE_TOKENS = [7, 8, 12, 13]

# Model defaults
DEFAULT_HIDDEN_SIZE = 100
DEFAULT_EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 40

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE_S0 = 0.001
DEFAULT_LEARNING_RATE_L0 = 0.0001
DEFAULT_LAMBDA = 0.01
DEFAULT_TAU = 1.0

# Length penalty constants
LENGTH_PENALTY_WEIGHT = 0.01

# REINFORCE temperature
REINFORCE_TEMPERATURE = 5.0

# RSA sampling alpha values
RSA_ALPHA_SAMPLE = 1.0
RSA_ALPHA_RSA = 0.0001
RSA_ALPHA_TEST = 0.0

# Image dimensions (ShapeWorld)
IMAGE_DIM = 64
IMAGE_X_MIN = 8
IMAGE_X_MAX = 48
IMAGE_ONE_QUARTER = (IMAGE_X_MAX - IMAGE_X_MIN) // 3
IMAGE_X_MIN_34 = IMAGE_X_MIN + IMAGE_ONE_QUARTER
IMAGE_X_MAX_34 = IMAGE_X_MAX - IMAGE_ONE_QUARTER
IMAGE_BUFFER = 10
IMAGE_SIZE_MIN = 3
IMAGE_SIZE_MAX = 8

# ShapeWorld constants
SHAPES = ['circle', 'square', 'rectangle', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
SHAPEWORLD_VOCAB = ['gray', 'shape', 'blue', 'square', 'circle', 'green', 
                     'red', 'rectangle', 'yellow', 'ellipse', 'white']
MAX_PLACEMENT_ATTEMPTS = 5

# Data split ratios
TRAIN_FRAC = 0.64
VAL_FRAC = 0.16
TEST_FRAC = 0.20

# Colors dataset
COLORS_IMAGE_SIZE = 64
COLORS_MIN_TOKEN_OCC = 2
COLORS_MAX_SENT_LEN = 16
COLORS_RANDOM_SEED = 42

