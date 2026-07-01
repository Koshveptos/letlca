

# map 

BOUNDARY = 1000.0          
MIN_START_DISTANCE = 150.0
MAX_START_DISTANCE = 1500.0

REACH_RADIUS = 30.0



### phisicssssss

DT = 0.05

MAX_SPEED = 300.0

MAX_STEPS = 800

MAX_DISTANCE = 2500.0

NOISE_STD = 2.0




## for pro 

TOTAL_TIMESTEPS = 1_000_000

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

N_STEPS = 2048

BATCH_SIZE = 64

N_EPOCHS = 10

CLIP_RANGE = 0.2

ENT_COEF = 0.01

VF_COEF = 0.5




#files

MODEL_PATH = "models/ppo_drone"

VEC_PATH = "models/vec_normalize.pkl"

LOG_DIR = "logs/"

CHECKPOINT_DIR = "models/checkpoints/"