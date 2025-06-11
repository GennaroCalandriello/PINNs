import numpy as np
import torch
import torch.nn as nn
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA')
else:
    device = torch.device('cpu')
    print('CPU')

#save name of string
os.makedirs("models", exist_ok=True)
os.makedirs("losses", exist_ok=True)

save_model = "models/pinn_model2d_norm.pth"
save_loss = "losses/pinn_loss2d_norm.npy"
use_only_first_layer_NTK = True # If True, NTK acts only on the first layer of NN, if you want to use all layers, ensure that the
# number of layers is about 30-40, for 500 neurons you should allocate ~1Tb on GPU!!!!!

log_NTK = True
update_lam = True
#hyperparameters
x_lb = -1.0
x_ub = 1.0
ylb = -1.0
yub = 1.0
t_lb = 0.0
t_ub = 1.0
nu = 0.01 # Wave constant
layers = [3, 400, 600, 600, 2] #Numerosi articoli usano pochi hidden layers, con 30-50 neuroni, che è la dimensione ideale per un kernel gradient
                               #descent performabile su tutti i layers. MA, se non metto almento 400-500 neuroni per layer, la rete è pessima.
kernel_size = 400
iterations = 40000 #epochs

scheduler_step = 2000 # Number of steps to update the learning rate
ntk_step = 100 # Number of steps to compute the NTK matrix

#■ it's a square!!-----------------------■
# bc1_coord  = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 1.0]])
# bc2_coord  = np.array([[0.0, 1.0, 0.0],[1.0, 1.0, 1.0]])
# bc3_coord  = np.array([[0.0, 0.0, 0.0],[1.0, 1.0, 0.0]])
# bc4_coord  = np.array([[0.0, 0.0, 1.0],[1.0, 1.0, 1.0]]) #|
#■---------------------------------------■

bc1_coord = np.array([[t_lb, x_lb, ylb], [t_ub, x_lb, yub]]) #left
bc2_coord = np.array([[t_lb, x_ub, ylb], [t_ub, x_ub, yub]]) #right
# bc3_coord = np.array([[t_lb, x_lb, yub], [t_ub, x_lb, yub]]) #top
# bc4_coord = np.array([[t_lb, x_ub, yub], [t_ub, x_ub, yub]]) #bottom
bc3_coord = np.array([[t_lb, x_lb, yub], [t_ub, x_ub, yub]])   # TOP: y fixed, x varies
bc4_coord = np.array([[t_lb, x_lb, ylb], [t_ub, x_ub, ylb]])   # BOTTOM: y fixed, x varies

dom_coord = np.array([[t_lb, x_lb, ylb], [t_ub, x_ub, yub]])
ics_coord = np.array([[t_lb, x_lb, ylb], [t_ub, x_ub, yub]])

"""Idee:
1. E se usassimo un simulatore CFD soltanto per le condizioni al contorno? Ciò dovrebbe  guidare la soluzione della PDE all'interno del dominio.
    Ricordo che questa cosa funziona con wave1D, dove prendo la soluzione analitica e la impongo su IC (qui t =0) e BC--EXPERIMENTING.
    
2. Posso usare tecniche proiettive per trasformare il problema in un problema di dimensione inferiore, come si fa con
   la decomposizione di Karhunen-Loeve, o con le tecniche di riduzione dimensionale come PCA.
   
   2.1 In Conformal Field Theory, si usa la proiezione su un insieme di funzioni ortogonali, come le funzioni di Bessel o le funzioni di Legendre.
   2.2 In Quantum Field Theory, si usa la proiezione su uno spazio di Hilbert, come lo spazio di Fock."""