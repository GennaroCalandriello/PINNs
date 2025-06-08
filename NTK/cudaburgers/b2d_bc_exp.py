from hyperpar import *
from bc import *

pde_bool = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Avoids KMP duplicate lib error

# References: [1] Is the Neural tangent kernel of PINNs deep learning general PDE always convergent? https://arxiv.org/abs/2412.06158
# [2] *When and why PINNs fail to train: A NTK perspective https://arxiv.org/pdf/2007.14527
# * code is inspired to this article, from repository: https://github.com/cemsoyleyici/PINN/blob/main
# The aim is to update the parameters lambda in each term of the loss function, transforming the problem
# of gradient descent into a Kernel Gradient Descent functional problem. How the matrix is constructed
# is explained in the article [2]. L = \lam_u L_u + \lam_ut L_ut + \lam_r L_r
#Qui eseguo una risoluzione di una 1d wave equation senza dati, solo con Ics e Bcs, se vuoi i dati sui residui 
#modifica la funzione r(x, a, c) e il sampler dei residui
#80000 epoche ottengo un errore assoluto massimo di 0.08 su un dominio [0,1]x[0,1] con a=0.5 e c=2


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
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, nu, kernel_size):
        
        # Normalization 
        # Assume res_sampler samples over [t, x, y]
        X, _ = res_sampler.sample(int(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]
        self.mu_y, self.sigma_y = self.mu_X[2], self.sigma_X[2]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # weights
        self.lam_ic_val = torch.tensor(1.0).float().to(device) # for initial conditions
        self.lam_bc_val = torch.tensor(1.0).float().to(device) # for boundary conditions
        self.lam_ru_val = torch.tensor(1.0).float().to(device) # for residuals u
        self.lam_rv_val = torch.tensor(1.0).float().to(device) # for residuals v
        # self.lam_ut_val = torch.tensor(1.0).float().to(device)
     
        
        # Wave constant
        self.nu = torch.tensor(nu).float().to(device)
        
        self.kernel_size = kernel_size # Size of the NTK matrix

        self.D1 = self.kernel_size    # boundary
        self.D2 = self.kernel_size    # ic   
        self.D3 = self.kernel_size    # residual  D1 = D3 = 3D2
        
        # Neural Network
        self.nn = NeuralNet(layers[0], layers[1], layers[-1]).to(device)

        self.optimizer_Adam = torch.optim.Adam(params=self.nn.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_Adam, gamma=0.9)
        
        # Logger
        self.loss_bcs_log = []
        self.loss_ic_log = []
        self.loss_ru_log = []
        self.loss_rv_log = []
        
        # NTK logger 
        self.K_ic_log = []
        self.K_bc_log = []
        self.K_ru_log = []
        self.K_rv_log = []
        
        # weights logger
        self.lam_bc_log = []
        self.lam_ic_log = []
        self.lam_ru_log = []
        self.lam_rv_log = []
    
    # Forward pass for u
    def net_uv(self, t, x, y):
            
        if x.dim()==0:
            x = x.reshape(1)
            y = y.reshape(1)
            t = t.reshape(1)
            uv = self.nn(torch.cat([t, x, y], dim=0))
        else:
            uv = self.nn(torch.cat([t, x, y], dim=1))
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    # Forward pass for the residual
    def net_r(self, t, x, y):
        u, v = self.net_uv(t, x, y)
        residual1, residual2 = self.operator(u,v, t, x, y,
                                 self.nu,
                                 self.sigma_t,
                                 self.sigma_x, self.sigma_y)
        return residual1, residual2
    
    # Gradient operation
    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, allow_unused=True)[0]
        return grad
    
    def compute_jacobian(self, output, params):
        
        if use_only_first_layer_NTK:
            
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
        
        else: ###!!!!WARNING: memory allocation increases exponentially with the number of parameters in the network
            output = output.reshape(-1)
            jacobians = []
            full_jacobian = None
            for p in params:
                # [output_dim, param_dim]
                jac_p, = torch.autograd.grad(
                    outputs=output, inputs=p,
                    grad_outputs=torch.eye(output.shape[0], device=output.device),
                    retain_graph=True, allow_unused=True, is_grads_batched=True
                )
                if jac_p is not None:
                    jacobians.append(jac_p.reshape(output.shape[0], -1))  # [N, param_dim]
            # Concatenate along param dimension: [N, total_param_dim]
            full_jacobian = torch.cat(jacobians, dim=1)
            return full_jacobian

    # Compute Neural Tangent Kernel's Trace Values
    def compute_ntk(self, J1, x1, J2, x2):
        
        d1 = J1[0].shape[0]
        d2 = J2[0].shape[0]
        # print("d1: ", d1, "d2: ", d2)
        Ker = torch.zeros((d1, d2), dtype=J1[0].dtype, device=J1[0].device)
        for i in range(len(J1)):
            K = torch.matmul(J1[i], J2[i].t())
            Ker = Ker + K
        return Ker


    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    """
    Scopo: campionare le BC - le funzioni get_*_bc_from_CFD(x) prende i punti random sulla frontiera, chiama la CFD “on the fly”, 
    e restituisce i valori BC aggiornati interpolando tra i valori della CFD."""
    
    def train(self, nIter=1000, batch_size=128, log_NTK=False, update_lam=False):
        
        # NTK
        self.nn.train()
        # CFD simulation
        
        for it in range(nIter):
            if it%10==0: print(f"Iteration {it+1}/{nIter}, burgers2d step")
            
            # Fetch boundary mini-batches
            X_ics_batch, uv_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size //5)
            X_bc1_batch, uv_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 5)
            X_bc2_batch, uv_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 5)
            X_bc3_batch, uv_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size // 5)
            X_bc4_batch, uv_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size // 5)
            # print("one value of X_ics_batch", X_ics_batch[0,:])ù
            # print("shape of X_ics_batch", X_ics_batch.shape) #shape is kernel_size//5
            # print(X_bc1_batch.shape, X_bc2_batch.shape, X_bc3_batch.shape, X_bc4_batch.shape)
            # print(X_bc1_batch[0:5,:])
            # Tensor
            X_ics_batch_tens = torch.tensor(X_ics_batch, requires_grad=True).float().to(device)
            uv_ics_batch_tens = torch.tensor(uv_ics_batch, requires_grad=True).float().to(device)
            X_bc1_batch_tens = torch.tensor(X_bc1_batch, requires_grad=True).float().to(device)
            X_bc2_batch_tens = torch.tensor(X_bc2_batch, requires_grad=True).float().to(device)
            X_bc3_batch_tens = torch.tensor(X_bc3_batch, requires_grad=True).float().to(device)
            X_bc4_batch_tens = torch.tensor(X_bc4_batch, requires_grad=True).float().to(device)
            uv_bc1_batch_tens = torch.tensor(uv_bc1_batch, requires_grad=True).float().to(device)
            uv_bc2_batch_tens = torch.tensor(uv_bc2_batch, requires_grad=True).float().to(device)
            uv_bc3_batch_tens = torch.tensor(uv_bc3_batch, requires_grad=True).float().to(device)
            uv_bc4_batch_tens = torch.tensor(uv_bc4_batch, requires_grad=True).float().to(device)
            
            u_pred_ics, v_pred_ics = self.net_uv(X_ics_batch_tens[:, 0:1], X_ics_batch_tens[:, 1:2], X_ics_batch_tens[:, 2:3])
            u_pred_bc1, v_pred_bc1 = self.net_uv(X_bc1_batch_tens[:, 0:1], X_bc1_batch_tens[:, 1:2], X_bc1_batch_tens[:, 2:3])
            u_pred_bc2, v_pred_bc2 = self.net_uv(X_bc2_batch_tens[:, 0:1], X_bc2_batch_tens[:, 1:2], X_bc2_batch_tens[:, 2:3])
            u_pred_bc3, v_pred_bc3 = self.net_uv(X_bc3_batch_tens[:, 0:1], X_bc3_batch_tens[:, 1:2], X_bc3_batch_tens[:, 2:3])
            u_pred_bc4, v_pred_bc4 = self.net_uv(X_bc4_batch_tens[:, 0:1], X_bc4_batch_tens[:, 1:2], X_bc4_batch_tens[:, 2:3])
            #PDE residuals
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size // 5)
            X_res_batch_tens = torch.tensor(X_res_batch, requires_grad=True).float().to(device)
            
            ru_pred, rv_pred = self.net_r(X_res_batch_tens[:, 0:1], X_res_batch_tens[:, 1:2], X_res_batch_tens[:, 2:3])
            # Compute the loss
            loss_ru = torch.mean(ru_pred ** 2)
            loss_rv = torch.mean(rv_pred ** 2)
            
            loss_bcs = torch.mean((uv_bc1_batch_tens[:, 0:1] - u_pred_bc1) ** 2) \
                    + torch.mean((uv_bc1_batch_tens[:, 1:2] - v_pred_bc1) ** 2) \
                    + torch.mean((uv_bc2_batch_tens[:, 0:1] - u_pred_bc2) ** 2) \
                    + torch.mean((uv_bc2_batch_tens[:, 1:2] - v_pred_bc2) ** 2) \
                    + torch.mean((uv_bc3_batch_tens[:, 0:1] - u_pred_bc3) ** 2) \
                    + torch.mean((uv_bc3_batch_tens[:, 1:2] - v_pred_bc3) ** 2) \
                    + torch.mean((uv_bc4_batch_tens[:, 0:1] - u_pred_bc4) ** 2) \
                    + torch.mean((uv_bc4_batch_tens[:, 1:2] - v_pred_bc4) ** 2)     
            loss_ics = torch.mean((uv_ics_batch_tens[:, 0:1] - u_pred_ics) ** 2) \
                    + torch.mean((uv_ics_batch_tens[:, 1:2] -v_pred_ics) ** 2)
                    
            loss = self.lam_ic_val * loss_ics +self.lam_bc_val * loss_bcs  +self.lam_ru_val*loss_ru +self.lam_rv_val*loss_rv
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
                self.loss_ic_log.append(loss_ics.detach().cpu().numpy())
                self.loss_ru_log.append(loss_ru.detach().cpu().numpy())
                self.loss_rv_log.append(loss_rv.detach().cpu().numpy())
                
                # self.loss_ut_ics_log.append(loss_ics_u_t.detach().cpu().numpy())
                print("Epoch:", it,"/", nIter)
                print('Loss: %.3e, Loss_bcs: %.3e, Loss_ics: %.3e, Loss_ru: %.3e, Loss_rv: %.3e' %
                      (loss.item(), loss_bcs, loss_ics, loss_ru, loss_rv))
                print(f'lambda_ic: {self.lam_ic_val:3e}')       
                print(f'lambda_bc: {self.lam_bc_val:3e}')
                print(f'lambda_ru: {self.lam_ru_val:3e}')
                print(f'lambda_rv: {self.lam_rv_val:3e}')
                print("Learning rate: ", self.my_lr_scheduler.get_last_lr()[0])
          
            if log_NTK:
                
                if it % ntk_step == 0:
                    print("Compute NTK...")
                    X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch, X_bc3_batch, X_bc4_batch])
                    X_ics_batch, uv_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size )
                    
                    
                    # Convert to the tensor
                    X_bc_batch_tens = torch.tensor(X_bc_batch, requires_grad=True).float().to(device)
                    X_ics_batch_tens = torch.tensor(X_ics_batch, requires_grad=True).float().to(device)
                    uv_ics_batch_tens = torch.tensor(uv_ics_batch, requires_grad=True).float().to(device)

                    # Get the parameters of NN
                    params = list(self.nn.parameters())
                    
                    # Store the trace 
                    K_bc_value = 0
                    K_ic_value = 0
                    K_ru_value = 0
                    K_rv_value = 0
                    # K_ruv_value = 0
                    
                    u_bc, v_bc = self.net_uv(X_bc_batch_tens[:,0:1], X_bc_batch_tens[:,1:2], X_bc_batch_tens[:,2:3])
                    bc_ntk_pred = torch.cat([u_bc, v_bc], dim=0)
                    u_ic, v_ic = self.net_uv(X_ics_batch_tens[:, 0:1], X_ics_batch_tens[:,1:2], X_ics_batch_tens[:,2:3])
                    ic_ntk_pred = torch.cat([u_ic, v_ic], dim=0)
                    res_ntk_u, res_ntk_v = self.net_r(X_res_batch_tens[:,0:1], X_res_batch_tens[:,1:2], X_res_batch_tens[:,2:3])
                    # u_t_ntk_pred = self.net_u_t(X_ics_batch_tens[:,0:1], X_ics_batch_tens[:,1:2])
                    # Jacobian of the neural networks
                    J_bc = self.compute_jacobian(bc_ntk_pred, params)
                    J_ic = self.compute_jacobian(ic_ntk_pred, params)
                    J_ru = self.compute_jacobian(res_ntk_u, params)
                    J_rv = self.compute_jacobian(res_ntk_v, params)
                    # J_ut = self.compute_jacobian(u_t_ntk_pred, params)
                    #on both:
                    # J_ruv = self.compute_jacobian(res_ntk_uv, params)
                    

                    # Neural tangent kernels of the neural networks / Trace values
                    K_bc_value = self.compute_ntk(J_bc, self.D1, J_bc, self.D1)
                    K_ic_value = self.compute_ntk(J_ic, self.D2, J_ic, self.D2)
                    K_ru_value = self.compute_ntk(J_ru, self.D3, J_ru, self.D3)
                    K_rv_value = self.compute_ntk(J_rv, self.D3, J_rv, self.D3)

                    # Convert tensor to numpy array
                    K_bc_value = K_bc_value.detach().cpu().numpy()
                    K_ic_value = K_ic_value.detach().cpu().numpy()
                    K_ru_value = K_ru_value.detach().cpu().numpy()
                    K_rv_value = K_rv_value.detach().cpu().numpy()
                    # K_ruv_value = K_ruv_value.detach().cpu().numpy()
                    
                    trace_K = np.trace(K_ic_value) + np.trace(K_bc_value)#+ np.trace(K_ruv_value)
  
                    # Store Trace values
                    self.K_bc_log.append(K_bc_value)
                    self.K_ic_log.append(K_ic_value)
                    self.K_ru_log.append(K_ru_value)
                    self.K_rv_log.append(K_rv_value)
                        
                    if update_lam:
                        self.lam_ic_val = trace_K / np.trace(K_ic_value)
                        self.lam_bc_val = trace_K / np.trace(K_bc_value)
                        self.lam_ru_val = trace_K / np.trace(K_ru_value)
                        self.lam_rv_val = trace_K / np.trace(K_rv_value)
                        # self.lam_ruv_val = trace_K / np.trace(K_ruv_value)
                        

                        # Store NTK weights
                        self.lam_bc_log.append(self.lam_bc_val)
                        self.lam_ic_log.append(self.lam_ic_val)
                        self.lam_ru_log.append(self.lam_ru_val)
                        self.lam_rv_log.append(self.lam_rv_val)
                        
                        # self.lam_ruv_log.append(self.lam_ruv_val)
                        
          
    # Evaluates predictions at test points
    def predict_uv(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        
        t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X_star[:, 2:3], requires_grad=True).float().to(device)

        self.nn.eval()

        u_star, v_star = self.net_uv(t, x, y)
        u_star = u_star.detach().cpu().numpy()
        v_star = v_star.detach().cpu().numpy()
        # uv_star = uv_star.detach().cpu().numpy()
        # u_star = uv_star[:, 0:1]
        # v_star = uv_star[:, 1:2]
        
        return u_star, v_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        
        
        t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X_star[:, 2:3], requires_grad=True).float().to(device)
        
        self.nn.eval()

        r_star = self.net_r(t, x)
        r_star = r_star.detach().cpu().numpy()
        return r_star

# Define the initial conditions
def u(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:,0:1]
    xc = x[:,1:2]
    yc = x[:,2:3]
    
    
    return np.sin(np.pi * xc) * np.cos( np.pi * yc)


def v(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:,0:1]
    xc = x[:,1:2]
    yc = x[:,2:3]
    
    return np.sin(np.pi * yc) * np.cos(np.pi * xc)

def u_bc(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:,0:1]
    xc = x[:,1:2]
    yc = x[:,2:3]
    
    
    return np.sin(np.pi * xc) * np.cos( np.pi * yc)* np.exp(-nu * np.pi**2 * tc)

def v_bc(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:,0:1]
    xc = x[:,1:2]
    yc = x[:,2:3]
    
    return np.sin(np.pi * yc) * np.cos(np.pi * xc)* np.exp(-nu * np.pi**2 * tc)

def r(u_tt):
    zerores = np.zeros((u_tt.shape[0], 1))
    return zerores

def operator(u, v, t, x, y, nu, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0):

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0] / sigma_y

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0] / sigma_t
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0] / sigma_x
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0] / sigma_y

    # Second-order derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0] / sigma_x
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0] / sigma_y

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0] / sigma_x
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0] / sigma_y

    # Residuals
    res_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)

    return res_u, res_v

zero_bc = lambda x: np.zeros((x.shape[0], 2))
ics_sampler = Sampler(3, ics_coord, lambda x: np.hstack([u(x, nu), v(x, nu)]), name='ICs')

res_sampler = Sampler(3, dom_coord, lambda x: np.zeros((x.shape[0],2)), name='Residuals')



#PINN model

burgers2d.setupB2d()  # Initialize the CFD simulation
burgers2d.mainB2d()

bc1_sampler = Sampler(3, bc1_coord, get_left_bc_from_history, name='BC1')
bc2_sampler = Sampler(3, bc2_coord, get_right_bc_from_history, name='BC2')
bc3_sampler = Sampler(3, bc3_coord, get_top_bc_from_history, name='BC3')
bc4_sampler = Sampler(3, bc4_coord, get_bottom_bc_from_history, name='BC4')
bcs_sampler = [bc1_sampler, bc2_sampler, bc3_sampler, bc4_sampler]

model = PINN(layers, operator, ics_sampler, bcs_sampler, res_sampler,nu, kernel_size)


def main():     
    train_bool = True
    #train
    if train_bool:
        print("Training the model...")
        
        

        model.train(iterations, batch_size=kernel_size, log_NTK=log_NTK, update_lam=update_lam)

        #save model
        torch.save(model.nn.state_dict(), save_model)
        print(f"Model saved as {save_model}")
        #save losses
        total_loss = []
        total_loss.append(model.loss_bcs_log)
        total_loss.append(model.loss_ic_log)
        np.save(save_loss, total_loss)
        print(f"Losses saved as {save_loss}")
        
def plot_loss():
    losses = np.load(save_loss, allow_pickle=True)
    loss_bcs = losses[0]
    loss_ru= losses[1]
    loss_rv = losses[2]
    loss_ic = losses[3]
    # loss_ruv = losses[4]

    plt.figure(figsize=(10, 6))
    plt.plot(loss_bcs, label='Loss BCS')
    plt.plot(loss_ru, label='Loss ru')
    plt.plot(loss_rv, label='Loss rv')
    plt.plot(loss_ic, label='Loss ICS')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.show()


def animate_plot_2d():
    import matplotlib.animation as animation

    # 1) Load model
    model = PINN(layers, operator, ics_sampler, bcs_sampler, res_sampler, nu, kernel_size)
    model.nn.load_state_dict(torch.load(save_model))

    # 2) Build evaluation grid (2D x-y, loop in time)
    Nx, Ny, Nt = 200, 200, 400
    t_lin = np.linspace(dom_coord[0,0], dom_coord[1,0], Nt)
    x_lin = np.linspace(dom_coord[0,1], dom_coord[1,1], Nx)
    y_lin = np.linspace(dom_coord[0,2], dom_coord[1,2], Ny)
    X, Y = np.meshgrid(x_lin, y_lin)  # both shape (Ny, Nx)

    # 3) Prepare storage for |velocity| frames
    U_frames = []
    with torch.no_grad():
        for i, t_val in enumerate(t_lin):
            # (Ny*Nx, 3): t, x, y
            pts = np.column_stack([
                np.full(X.size, t_val), X.ravel(), Y.ravel()
            ])
            u_pred, v_pred = model.predict_uv(pts)  # shape (Ny*Nx, 2)
            u_pred = u_pred.reshape(Ny, Nx)
            v_pred = v_pred.reshape(Ny, Nx)
            mag = np.sqrt(u_pred**2 + v_pred**2)  # shape (Ny, Nx)
            U_frames.append(mag)

    # 4) Setup the plot
    fig, ax = plt.subplots(figsize=(6,5))
    vmin = np.min(U_frames)
    vmax = np.max(U_frames)
    cax = ax.pcolormesh(X, Y, U_frames[0], cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    title = ax.set_title(f'time = {t_lin[0]:.3f}')
    fig.colorbar(cax, ax=ax, label=r'$|\vec{u}|$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 5) Update function
    def update(frame):
        cax.set_array(U_frames[frame].ravel())
        title.set_text(f'time = {t_lin[frame]:.3f}')
        return cax, title

    # 6) Animation
    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=50, blit=False, repeat=True
    )
    plt.show()
    return ani

    
if __name__ == '__main__':
                
    #load the model
    # main()
    # plot_loss()
    # plot()
    animate_plot_2d()
    pass
#plot analytical initial conditions




                    
        