import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Avoids KMP duplicate lib error

import numpy as np

from scipy.interpolate import griddata

import matplotlib.pyplot as plt

# References: [1] Is the Neural tangent kernel of PINNs deep learning general PDE always convergent? https://arxiv.org/abs/2412.06158
# [2] *When and why PINNs fail to train: A NTK perspective https://arxiv.org/pdf/2007.14527
# * code is inspired to this article, from repository: https://github.com/cemsoyleyici/PINN/blob/main
# The aim is to update the parameters lambda in each term of the loss function, transforming the problem
# of gradient descent into a Kernel Gradient Descent functional problem. How the matrix is constructed
# is explained in the article [2]. L = \lam_u L_u + \lam_ut L_ut + \lam_r L_r
#Qui eseguo una risoluzione di una 1d wave equation senza dati, solo con Ics e Bcs, se vuoi i dati sui residui 
#modifica la funzione r(x, a, c) e il sampler dei residui
#80000 epoche ottengo un errore assoluto massimo di 0.08 su un dominio [0,1]x[0,1] con a=0.5 e c=2

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA')
else:
    device = torch.device('cpu')
    print('CPU')

#hyperparameters
x_lb = 0.0
x_ub = 1.0
ylb = 0.0
yub = 1.0
t_lb = 0.0
t_ub = 1.0

ics_coord = np.array([[0.0,0.0],[0.0, x_ub]])
bc1_coord = np.array([[0.0, 0.0], [t_ub, 0.0]])
bc2_coord = np.array([[0.0, x_ub], [t_ub, x_ub]])
print(ics_coord[0:1, :])
print(ics_coord[1:2, :])

dom_coord = np.array([[t_lb, x_lb], [t_ub, x_ub]])
a = 0.5
c = 2
scheduler_step = 1000 # Number of steps to update the learning rate
ntk_step = 100 # Number of steps to compute the NTK matrix
class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name = None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
    def sample(self, N):
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y

# Creating Neural Network
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        # initialize weight and Biases
        nn.init.xavier_uniform_(self.l1.weight)
        self.l1.bias.data.fill_(0.0)

        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l2.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l2.weight)

        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l3.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l3.weight)

        self.l4 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.l4.weight)
        self.l4.bias.data.fill_(0.0)
        
    
    def forward(self, x):
        out = self.l1(x)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        out = self.tanh(out)
        out = self.l4(out)
        return out

class PINN:
    # Initialize the class
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size):
        
        # Normalization 
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # weights
        self.lam_u_val = torch.tensor(1.0).float().to(device)
        self.lam_ut_val = torch.tensor(1.0).float().to(device)
        self.lam_r_val = torch.tensor(1.0).float().to(device)
        
        # Wave constant
        self.c = torch.tensor(c).float().to(device)
        
        self.kernel_size = kernel_size # Size of the NTK matrix

        self.D1 = self.kernel_size    # boundary
        self.D2 = self.kernel_size    # ut   
        self.D3 = self.kernel_size    # residual  D1 = D3 = 3D2
        
        # Neural Network
        self.nn = NeuralNet(layers[0], layers[1], layers[-1]).to(device)

        self.optimizer_Adam = torch.optim.Adam(params=self.nn.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_Adam, gamma=0.9)
        
        # Logger
        self.loss_bcs_log = []
        self.loss_ut_ics_log = []
        self.loss_res_log = []
         
        # NTK logger 
        self.K_u_log = []
        self.K_ut_log = []
        self.K_r_log = []
        
        # weights logger
        self.lam_u_log = []
        self.lam_ut_log = []
        self.lam_r_log = [] 
    
    # Forward pass for u
    def net_u(self, t, x):
            
        if x.dim()==0:
            x = x.reshape(1)
            t = t.reshape(1)
            u = self.nn(torch.cat([t, x], dim=0))
        else:
            u = self.nn(torch.cat([t, x], dim=1))
        return u

    # Forward pass for du/dt
    def net_u_t(self, t, x):

        u = self.net_u(t, x)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0] / self.sigma_t
        return u_t

    # Forward pass for the residual
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.c,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual
    
    # Gradient operation
    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, allow_unused=True)[0]
        return grad

    # Compute Jacobian for each weights and biases in each layer and return a list 
    def compute_jacobian(self, output, params):
        output = output.reshape(-1)
        J_dum = []
        J_List = []
        
        for i in range(len(params)):
            grad, = torch.autograd.grad(output, params[i], (torch.eye(output.shape[0]).to(device),),retain_graph=True, allow_unused=True, is_grads_batched=True)
            if grad == None:
                pass
            else:
                J_dum.append(grad)

                if np.mod(i,2)==1:
                    if grad == None:
                        pass
                    J_List.append(torch.cat((J_dum[i-1].flatten().reshape(len(output),-1),grad.flatten().reshape(len(output),-1)), 1))
        return J_List

    # Compute Neural Tangent Kernel's Trace Values
    def compute_ntk(self, J1, d1, J2, d2):
        
        Ker = torch.zeros((d1,d2)).float().to(device)

        for i in range(len(J1)):
            K = torch.matmul(J1[i], J2[i].t())
            Ker = Ker + K       
        return Ker


    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=False, update_lam=False):
        
        # NTK
        self.nn.train()
        
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 3)
            X_bc1_batch, _ = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 3)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 3)
            
            # Tensor
            X_ics_batch_tens = torch.tensor(X_ics_batch, requires_grad=True).float().to(device)
            u_ics_batch_tens = torch.tensor(u_ics_batch, requires_grad=True).float().to(device)
            X_bc1_batch_tens = torch.tensor(X_bc1_batch, requires_grad=True).float().to(device)
            X_bc2_batch_tens = torch.tensor(X_bc2_batch, requires_grad=True).float().to(device)
            
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)
            X_res_batch_tens = torch.tensor(X_res_batch, requires_grad=True).float().to(device)
            
            u_pred_ics = self.net_u(X_ics_batch_tens[:, 0:1], X_ics_batch_tens[:, 1:2])
            u_t_pred_ics = self.net_u_t(X_ics_batch_tens[:, 0:1], X_ics_batch_tens[:, 1:2])
            u_pred_bc1 = self.net_u(X_bc1_batch_tens[:, 0:1], X_bc1_batch_tens[:, 1:2])
            u_pred_bc2 = self.net_u(X_bc2_batch_tens[:, 0:1], X_bc2_batch_tens[:, 1:2])
            r_pred = self.net_r(X_res_batch_tens[:, 0:1], X_res_batch_tens[:, 1:2])
            
            loss_bcs = torch.mean((u_ics_batch_tens - u_pred_ics) ** 2) + torch.mean(u_pred_bc1 ** 2) + torch.mean(u_pred_bc2 ** 2)
            loss_ics_u_t = torch.mean(u_t_pred_ics ** 2)
            loss_res = torch.mean(r_pred ** 2)

            loss = self.lam_r_val * loss_res + self.lam_u_val * loss_bcs + self.lam_ut_val * loss_ics_u_t 
            
            # Backward and optimize 
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            if it % scheduler_step == 0:
                self.my_lr_scheduler.step()
            
            # Print
            if it % ntk_step == 0:

                # Store losses
                self.loss_bcs_log.append(loss_bcs.detach().cpu().numpy())
                self.loss_res_log.append(loss_res.detach().cpu().numpy())
                self.loss_ut_ics_log.append(loss_ics_u_t.detach().cpu().numpy())

                print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e' %
                      (it, loss.item(), loss_res, loss_bcs, loss_ics_u_t))
                
                print(f'lambda_u: {self.lam_u_val:3e}')
                print(f'lambda_ut: {self.lam_ut_val:3e}')
                print(f'lambda_r: {self.lam_r_val:3e}')
          
            if log_NTK:
                
                if it % 50 == 0:
                    print("Compute NTK...")
                    X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch])
                    X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size )
                    
                    # Convert to the tensor
                    X_bc_batch_tens = torch.tensor(X_bc_batch, requires_grad=True).float().to(device)
                    X_ics_batch_tens = torch.tensor(X_ics_batch, requires_grad=True).float().to(device)
                    u_ics_batch_tens = torch.tensor(u_ics_batch, requires_grad=True).float().to(device)

                    # Get the parameters of NN
                    params = list(self.nn.parameters())
                    
                    # Store the trace 
                    K_u_value = 0
                    K_ut_value = 0
                    K_r_value = 0
                    
                    u_ntk_pred = self.net_u(X_bc_batch_tens[:,0:1], X_bc_batch_tens[:,1:2])
                    u_t_ntk_pred = self.net_u_t(X_ics_batch_tens[:,0:1], X_ics_batch_tens[:,1:2])
                    r_ntk_pred = self.net_r(X_res_batch_tens[:,0:1], X_res_batch_tens[:,1:2])

                    # Jacobian of the neural networks
                    J_u = self.compute_jacobian(u_ntk_pred, params)
                    J_ut = self.compute_jacobian(u_t_ntk_pred, params)
                    J_r = self.compute_jacobian(r_ntk_pred, params)

                    # Neural tangent kernels of the neural networks / Trace values
                    K_u_value = self.compute_ntk(J_u, self.D1, J_u, self.D1)
                    K_ut_value = self.compute_ntk(J_ut, self.D2, J_ut, self.D2)
                    K_r_value = self.compute_ntk(J_r, self.D3, J_r, self.D3)

                    # Convert tensor to numpy array
                    K_u_value = K_u_value.detach().cpu().numpy()
                    K_ut_value = K_ut_value.detach().cpu().numpy()
                    K_r_value = K_r_value.detach().cpu().numpy()
                    
                    trace_K = np.trace(K_u_value) + np.trace(K_ut_value) + np.trace(K_r_value)
  
                    # Store Trace values
                    self.K_u_log.append(K_u_value)
                    self.K_ut_log.append(K_ut_value)
                    self.K_r_log.append(K_r_value)
                        
                    if update_lam:

                        self.lam_u_val = trace_K / np.trace(K_u_value)
                        self.lam_ut_val = trace_K / np.trace(K_ut_value)
                        self.lam_r_val = trace_K / np.trace(K_r_value)

                        # Store NTK weights
                        self.lam_u_log.append(self.lam_u_val)
                        self.lam_ut_log.append(self.lam_ut_val)
                        self.lam_r_log.append(self.lam_r_val)
          
    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        
        t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)

        self.nn.eval()

        u_star = self.net_u(t, x)
        u_star = u_star.detach().cpu().numpy()
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        
        t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
        
        self.nn.eval()

        r_star = self.net_r(t, x)
        r_star = r_star.detach().cpu().numpy()
        return r_star

# Define the exact solution and its derivatives
def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
            a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)

def u_t(x,a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_t = -  c * np.pi * np.sin(np.pi * x) * np.sin(c * np.pi * t) - \
            a * 4 * c * np.pi * np.sin(2 * c * np.pi* x) * np.sin(4 * c * np.pi * t)
    return u_t

def u_tt(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
            a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return u_tt

def u_xx(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
              a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return  u_xx

def r(x, a, c):
    u_tt1 = u_tt(x, a, c)
    zerores = np.zeros_like(u_tt1)
    return zerores

def operator(u, t, x, c, sigma_t=1.0, sigma_x=1.0):

    u_t = torch.autograd.grad(
        u, t, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0] / sigma_t
    
    u_tt = torch.autograd.grad(
        u_t, t, 
        grad_outputs=torch.ones_like(u_t),
        retain_graph=True,
        create_graph=True
    )[0] / sigma_t
    
    u_x = torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0] / sigma_x

    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0] / sigma_x     
    
    residual = u_tt - c**2 * u_xx
    return residual
                    

#domain boundaries   
ics_sampler = Sampler(2, ics_coord, lambda x: u(x, a, c), name='ICs')
bc1_sampler = Sampler(2, bc1_coord, lambda x: u(x, a, c), name='BC1')
bc2_sampler = Sampler(2, bc2_coord, lambda x: u(x, a, c), name='BC2')

res_sampler = Sampler(2, dom_coord, lambda x: r(x, a, c), name='Residuals')
bcs_sampler = [bc1_sampler, bc2_sampler]

#PINN model


layers = [2, 500, 500, 500, 1]
kernel_size = 150
model = PINN(layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size)

def main():     
    train_bool = True
    #train
    if train_bool:
        print("Training the model...")
        iterations = 40000
        log_NTK = True
        update_lam = True

        model.train(iterations, batch_size=kernel_size, log_NTK=log_NTK, update_lam=update_lam)

        #save model
        torch.save(model.nn.state_dict(), 'pinn_model.pth')
        print("Model saved as 'pinn_model.pth'")
        #save losses
        total_loss = []
        total_loss.append(model.loss_bcs_log)
        total_loss.append(model.loss_ut_ics_log)
        total_loss.append(model.loss_res_log)
        np.save('losses.npy', total_loss)
        print("Losses saved as 'losses.npy'")
        
        
        
def plot():
        model = PINN(layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size)
        model.nn.load_state_dict(torch.load('pinn_model.pth'))
        sample_size = 200
        t = np.linspace(dom_coord[0, 0], dom_coord[1, 0], sample_size)[:, None]
        x = np.linspace(dom_coord[0, 1], dom_coord[1, 1], sample_size)[:, None]
        t, x = np.meshgrid(t, x)
        X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))


        u_star = u(X_star, a, c)
        r_star = r(X_star, a, c)   

        #predict u
        u_pred = model.predict_u(X_star)
        r_pred = model.predict_r(X_star)
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)  

        #plot 
        U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
        U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')

        plt.figure(figsize=(18, 9))
        plt.subplot(2, 3, 1)
        plt.pcolor(t, x, U_star, cmap='jet', shading='auto')
        plt.colorbar()
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Exact u(t, x)')
        plt.tight_layout()

        plt.subplot(2, 3, 2)
        plt.pcolor(t, x, U_pred, cmap='jet', shading='auto')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Predicted u(t, x)')
        plt.tight_layout()
        
        plt.subplot(2, 3, 3)
        plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet', shading='auto')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Absolute error')
        plt.tight_layout()

def plot_loss():
    losses = np.load('losses.npy', allow_pickle=True)
    loss_bcs = losses[0]
    loss_ut_ics = losses[1]
    loss_res = losses[2]

    plt.figure(figsize=(10, 6))
    plt.plot(loss_bcs, label='Loss BCS')
    plt.plot(loss_ut_ics, label='Loss UT ICS')
    plt.plot(loss_res, label='Loss Residual')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.show()

def animate_plot():
    import matplotlib.animation as animation
    # 1) load model
    model = PINN(layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size)
    model.nn.load_state_dict(torch.load('pinn_model.pth'))
    

    # 2) build evaluation grid
    Nx = 300
    Nt = 300
    t_lin = np.linspace(dom_coord[0,0], dom_coord[1,0], Nt)
    x_lin = np.linspace(dom_coord[0,1], dom_coord[1,1], Nx)
    T, X = np.meshgrid(t_lin, x_lin)            # both shape (Nx, Nt)
    pts = np.column_stack([T.ravel(), X.ravel()])  # (Nx*Nt, 2)

    # 3) predict and reshape
    with torch.no_grad():
        u_pred = model.predict_u(pts)           # assume returns shape (Nx*Nt,1)
    U = griddata(pts, u_pred.ravel(), (T, X), method='cubic')
    U_exact = griddata(pts, u(pts, a, c).ravel(), (T, X), method='cubic')  # exact solution
    # U now shape (Nx, Nt)

    # 4) set up the figure
    fig, ax = plt.subplots(figsize=(8,4))
    line, = ax.plot(x_lin, U[:,0], lw=2)
    line_exact, = ax.plot(x_lin, U_exact[:,0], lw=2, linestyle='--', color='orange')
    title = ax.text(0.5,1.05, '', transform=ax.transAxes, ha='center')
    ax.set_xlim(x_lin[0], x_lin[-1])
    ax.set_ylim(np.min(U), np.max(U))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    plt.legend(['Predicted u(t,x)', 'Exact u(t,x)'], loc='upper right')
    #grid
    ax.grid(True)
    plt.tight_layout()

    # 5) update function
    def update(frame):
        line.set_ydata(U[:, frame])
        line_exact.set_ydata(U_exact[:, frame])
        title.set_text(f'time = {t_lin[frame]:.3f}')
        return line, line_exact, title

    # 6) animation
    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=50, blit=True, repeat=True
    )
    plt.show()
    ani.save("wave 1D.gif", writer='pillow', fps=30, dpi=100)
    return ani
    
if __name__ == '__main__':
                
    #load the model
    # main()
    # plot_loss()
    # plot()
    animate_plot()
      
                    
        