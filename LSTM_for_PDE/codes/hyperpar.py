import numpy as np

LAYERS = [2, 60, 60 , 60, 1]
NU = 0.0005
LR = 0.001
EPOCHS = 200
N_INT, N_BC, N_IC = 1000, 300, 300
T_max = 0.5
SEQ_LEN = 20
NX = 400
NT = 200
DT = 0.001
X_LB, X_UB = -1, 1
Y_LB, Y_UB = X_LB, X_UB
Z_LB, Z_UB = X_LB, X_UB
T_LB, T_UB = 0.0, 1.0
cx, cy, r = 0.0, 0.0, 0.2 #circle center and radius

REYNOLDS = 1000 #Reynolds number, generally 1000-2000, 1000 for simple problems
HIDDEN_SIZE = 50 #LSTM hidden size, generally 50-100, 100 for complex problems
NUM_LAYERS = 2 #LSTM layers, generally 2-4, 3 for complex problems more layers learn more temporal patterns but train slowe
N_SAMPLES = 1000
N_COLLOCATION = 1000 #number of collocation points, generally 1000-10000, 10000 for complex problems
N_OBSTACLE = 400#number of obstacle points, generally 1000-10000, 1000 for simple problems
N_WALLS = 400 #number of wall points, generally 1000-10000, 1000 for simple problems

#lambda values for loss functions in LSTM
LAMBDA_DATA = 1.0 #weight for data loss
LAMBDA_PDE = 1 #weight for PDE loss
LAMBDA_BC = 1.0 #weight for BC loss
LAMBDA_IC = 1.0 #weight for IC loss
LAMBDA_OBS = 1.0 #weight for observation loss

BATCH_SIZE = 128 #batch size for training, generally 32-128, 64 for complex problems

###============================INLET BC============================
# number of inlet BC points
N_INLET    = 300
N_OUTLET   = 300
# weight for inlet BC loss
LAMBDA_INLET = 1.0
LAMBDA_OUTLET = 1.0
# prescribed inlet velocity (uniform in x‐direction)
U_INLET    = 0.3
x_ub_inlet = 0.3
x_lb_inlet = 0.0


SAVE_U = False
SAVE_LOSS = True
SAVE_MODEL = True