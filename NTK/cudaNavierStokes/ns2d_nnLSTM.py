from hyperpar import *
from bc import *

pde_bool = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # Avoids KMP duplicate lib error


# References: [1] Is the Neural tangent kernel of PINNs deep learning general PDE always convergent? https://arxiv.org/abs/2412.06158
# [2] *When and why PINNs fail to train: A NTK perspective https://arxiv.org/pdf/2007.14527
# * code is inspired to this article, from repository: https://github.com/cemsoyleyici/PINN/blob/main
# The aim is to update the parameters lambda in each term of the loss function, transforming the problem
# of gradient descent into a Kernel Gradient Descent functional problem. How the matrix is constructed
# is explained in the article [2]. L = \lam_u L_u + \lam_ut L_ut + \lam_r L_r
# Qui eseguo una risoluzione di una 1d wave equation senza dati, solo con Ics e Bcs, se vuoi i dati sui residui
# modifica la funzione r(x, a, c) e il sampler dei residui
# 80000 epoche ottengo un errore assoluto massimo di 0.08 su un dominio [0,1]x[0,1] con a=0.5 e c=2


# Define the initial conditions
def u_ic_torch(x, y):
    return torch.sin(np.pi * x) * torch.cos(np.pi * y)


def v_ic_torch(x, y):
    return torch.sin(np.pi * y) * torch.cos(np.pi * x)


def obstacle_mask(x, y, xc=0.0, yc=0.0, r=0.0):
    # x, y shape: (batch, 1)
    return ((x - xc) ** 2 + (y - yc) ** 2 > r**2).float()


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):

        x = self.coords[0:1, :] + (
            self.coords[1:2, :] - self.coords[0:1, :]
        ) * np.random.rand(N, self.dim)
        y = self.func(x)

        return x, y


class BCSampler:
    """
    Sampler for boundary conditions: samples uniformly on a hyperplane (boundary)
    by fixing one or more coordinates and randomizing the others.
    """

    def __init__(self, dim, coords, func, fixed_dims, fixed_values, name=None):
        """
        dim:         Number of dimensions (e.g. 3 for t, x, y)
        coords:      Array of shape (2, dim), giving min and max for each dim
        func:        Function to evaluate at sampled points
        fixed_dims:  List of indices to fix (e.g. [2] for y-boundary)
        fixed_values: List/array of values to fix at those indices
        name:        Optional name for sampler
        """
        self.dim = dim
        self.coords = coords
        self.func = func
        self.fixed_dims = fixed_dims
        self.fixed_values = np.array(fixed_values)
        self.name = name

    def sample(self, N):
        # Uniform random sampling for all dimensions
        x = self.coords[0:1, :] + (
            self.coords[1:2, :] - self.coords[0:1, :]
        ) * np.random.rand(N, self.dim)
        # Fix specified boundary coordinates
        for i, d in enumerate(self.fixed_dims):
            x[:, d] = self.fixed_values[i]
        y = self.func(x)

        return x, y


# Creating Neural Network
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(hidden_size, hidden_size)
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
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=len(layers) - 2, batch_first=True
        )

    def forward(self, x):
        # x: (batch, 3): t, x, y
        t = x[:, 0:1]
        xx = x[:, 1:2]
        yy = x[:, 2:3]
        # Rete "core"
        seq = x.unsqueeze(1)  # (batch, 1, 3)

        with torch.backends.cudnn.flags(enabled=False):
            h, _ = self.rnn(seq)  # (batch, 1, hidden_size)

        out = h.squeeze(1)  # (batch, hidden_size)
        out = self.l1(out)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        out = self.tanh(out)
        out = self.l4(out)  # (batch, 3): [u_net, v_net, p_net]

        # Calcola la maschera dell'ostacolo: 1 fluido, 0 solido
        mask = obstacle_mask(xx, yy)  # shape (batch, 1) o (batch,)

        # IC nel fluido, zero (o BC solida) nell'ostacolo
        u_ic = u_ic_torch(xx, yy)
        v_ic = v_ic_torch(xx, yy)
        p_ic = torch.zeros_like(xx)  # o p_ic_torch(xx, yy) * mask, se vuoi
        # Se t=0, la rete restituisce direttamente l’IC con l’ostacolo encode-ato

        u = u_ic + out[:, 0:1]
        v = v_ic + out[:, 1:2]
        p = p_ic + out[:, 2:3]

        return torch.cat([u, v, p], dim=1)  # (batch, 2)


class PINN:
    # Initialize the class
    def __init__(
        self, layers, operator, ics_sampler, bcs_sampler, res_sampler, nu, kernel_size
    ):

        # Normalization
        # Assume res_sampler samples over [t, x, y]
        X, _ = res_sampler.sample(int(1e5))
        # self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        # self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        # self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]
        # self.mu_y, self.sigma_y = self.mu_X[2], self.sigma_X[2]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # weights
        self.lam_bc_val = (
            torch.tensor(1.0).float().to(device)
        )  # for boundary conditions
        self.lam_ic_val = torch.tensor(1.0).float().to(device)  # for initial conditions
        self.lam_ru_val = torch.tensor(1.0).float().to(device)  # for residuals u
        self.lam_rv_val = torch.tensor(1.0).float().to(device)  # for residuals v
        self.lam_div_val = torch.tensor(1.0).float().to(device)  # for divergence

        # Wave constant
        self.nu = torch.tensor(nu).float().to(device)

        self.kernel_size = kernel_size  # Size of the NTK matrix

        self.D1 = self.kernel_size  # boundary
        self.D2 = self.kernel_size  # ic
        self.D3 = self.kernel_size  # residual  D1 = D3 = 3D2

        # Neural Network
        self.nn = NeuralNet(layers[0], layers[1], layers[-1]).to(device)

        self.optimizer_Adam = torch.optim.Adam(
            params=self.nn.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_Adam, gamma=0.9
        )

        # Logger
        self.loss_bcs_log = []
        self.loss_ic_log = []
        self.loss_ru_log = []
        self.loss_rv_log = []
        self.loss_div_log = []
        self.loss_total_log = []

        # NTK logger
        self.K_ic_log = []
        self.K_bc_log = []
        self.K_ru_log = []
        self.K_rv_log = []
        self.K_div_log = []

        # weights logger
        self.lam_bc_log = []
        self.lam_ic_log = []
        self.lam_ru_log = []
        self.lam_rv_log = []
        self.lam_div_log = []

    # Forward pass for u
    def net_uvp(self, t, x, y):

        if x.dim() == 0:
            x = x.reshape(1)
            y = y.reshape(1)
            t = t.reshape(1)
            uvp = self.nn(torch.cat([t, x, y], dim=0))
        else:
            uvp = self.nn(torch.cat([t, x, y], dim=1))
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        return u, v, p

    # Forward pass for the residual
    def net_r(self, t, x, y):
        u, v, p = self.net_uvp(t, x, y)
        residual1, residual2, resudial3 = self.operator(u, v, p, t, x, y, self.nu)

        return residual1, residual2, resudial3

    # Gradient operation
    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(
            y, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=True
        )[0]
        return grad

    def compute_jacobian(self, output, params):

        if (
            use_only_first_layer_NTK
        ):  # Here we use only the first layer of the NN to compute the NTK matrix, this is useful to reduce memory allocation

            output = output.reshape(-1)
            J_dum = []
            J_List = []

            for i in range(len(params)):
                (grad,) = torch.autograd.grad(
                    output,
                    params[i],
                    (torch.eye(output.shape[0]).to(device),),
                    retain_graph=True,
                    allow_unused=True,
                    is_grads_batched=True,
                )
                if grad == None:
                    pass
                else:
                    J_dum.append(grad)

                    if np.mod(i, 2) == 1:
                        if grad == None:
                            pass
                        J_List.append(
                            torch.cat(
                                (
                                    J_dum[i - 1].flatten().reshape(len(output), -1),
                                    grad.flatten().reshape(len(output), -1),
                                ),
                                1,
                            )
                        )
            return J_List

        else:  ###!!!!WARNING: memory allocation increases exponentially with the number of parameters in the network
            output = output.reshape(-1)
            jacobians = []
            full_jacobian = None
            for p in params:
                # [output_dim, param_dim]
                (jac_p,) = torch.autograd.grad(
                    outputs=output,
                    inputs=p,
                    grad_outputs=torch.eye(output.shape[0], device=output.device),
                    retain_graph=True,
                    allow_unused=True,
                    is_grads_batched=True,
                )
                if jac_p is not None:
                    jacobians.append(
                        jac_p.reshape(output.shape[0], -1)
                    )  # [N, param_dim]
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
        # print("before normalization X values:", X[:5])

        # print("after normalization X values:", X[:5])
        return X, Y

    # Trains the model by minimizing the MSE loss

    def normalize_loss(self, prediction, target):
        """
        non normalizza proprio un cazz
        """
        mse = torch.mean((prediction - target) ** 2)

        return mse

    def train(self, nIter=1000, batch_size=128, log_NTK=False, update_lam=False):
        """Scopo: tramite PyBind11 runno una simulazione CFD su CUDA, interpolo i risultati su tutti i valori u, v per
        ogni istante temporale, trovo il valore più vicino a quello estratto dalla rete, e lo uso come ground truth.
        Qui sto sperimentando una normalizzazione delle BCs per renderle adimensionali
        """

        # NTK
        self.nn.train()
        # CFD simulation

        for it in range(nIter):
            if it % 10 == 0:
                print(f"Iteration {it}/{nIter}")

            X_ics_batch, uv_ics_batch = self.fetch_minibatch(
                self.ics_sampler, batch_size
            )
            X_bc1_batch, uv_bc1_batch = self.fetch_minibatch(
                self.bcs_sampler[0], batch_size // 4
            )
            X_bc2_batch, uv_bc2_batch = self.fetch_minibatch(
                self.bcs_sampler[1], batch_size // 4
            )
            X_bc3_batch, uv_bc3_batch = self.fetch_minibatch(
                self.bcs_sampler[2], batch_size // 4
            )
            X_bc4_batch, uv_bc4_batch = self.fetch_minibatch(
                self.bcs_sampler[3], batch_size // 4
            )

            # Tensor conversion
            X_ics_batch_tens = (
                torch.tensor(X_ics_batch, requires_grad=True).float().to(device)
            )
            X_bc1_batch_tens = (
                torch.tensor(X_bc1_batch, requires_grad=True).float().to(device)
            )
            X_bc2_batch_tens = (
                torch.tensor(X_bc2_batch, requires_grad=True).float().to(device)
            )
            X_bc3_batch_tens = (
                torch.tensor(X_bc3_batch, requires_grad=True).float().to(device)
            )
            X_bc4_batch_tens = (
                torch.tensor(X_bc4_batch, requires_grad=True).float().to(device)
            )
            uv_ics_batch_tens = (
                torch.tensor(uv_ics_batch, requires_grad=True).float().to(device)
            )
            uv_bc1_batch_tens = (
                torch.tensor(uv_bc1_batch, requires_grad=True).float().to(device)
            )
            uv_bc2_batch_tens = (
                torch.tensor(uv_bc2_batch, requires_grad=True).float().to(device)
            )
            uv_bc3_batch_tens = (
                torch.tensor(uv_bc3_batch, requires_grad=True).float().to(device)
            )
            uv_bc4_batch_tens = (
                torch.tensor(uv_bc4_batch, requires_grad=True).float().to(device)
            )

            # print("some values of X_bc1_batch_tens:", X_bc1_batch_tens[:5])
            # print("some values of uv_bc1_batch_tens:", uv_bc1_batch_tens[:5])
            # print("some values of X_bc2_batch_tens:", X_bc2_batch_tens[:5])
            # print("some values of uv_bc2_batch_tens:", uv_bc2_batch_tens[:5])
            # print("some values of X_bc3_batch_tens:", X_bc3_batch_tens[:5])
            # print("some values of uv_bc3_batch_tens:", uv_bc3_batch_tens[:5])

            # Compute the predictions for boundary conditions
            u_pred_ics, v_pred_ics, _ = self.net_uvp(
                X_ics_batch_tens[:, 0:1],
                X_ics_batch_tens[:, 1:2],
                X_ics_batch_tens[:, 2:3],
            )
            u_pred_bc1, v_pred_bc1, _ = self.net_uvp(
                X_bc1_batch_tens[:, 0:1],
                X_bc1_batch_tens[:, 1:2],
                X_bc1_batch_tens[:, 2:3],
            )
            u_pred_bc2, v_pred_bc2, _ = self.net_uvp(
                X_bc2_batch_tens[:, 0:1],
                X_bc2_batch_tens[:, 1:2],
                X_bc2_batch_tens[:, 2:3],
            )
            u_pred_bc3, v_pred_bc3, _ = self.net_uvp(
                X_bc3_batch_tens[:, 0:1],
                X_bc3_batch_tens[:, 1:2],
                X_bc3_batch_tens[:, 2:3],
            )
            u_pred_bc4, v_pred_bc4, _ = self.net_uvp(
                X_bc4_batch_tens[:, 0:1],
                X_bc4_batch_tens[:, 1:2],
                X_bc4_batch_tens[:, 2:3],
            )

            # PDE residuals
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)
            X_res_batch_tens = (
                torch.tensor(X_res_batch, requires_grad=True).float().to(device)
            )

            ru_pred, rv_pred, rdiv_pred = self.net_r(
                X_res_batch_tens[:, 0:1],
                X_res_batch_tens[:, 1:2],
                X_res_batch_tens[:, 2:3],
            )

            # Compute the pde losses (not normalized)
            loss_ru = torch.mean(ru_pred**2)
            loss_rv = torch.mean(rv_pred**2)
            loss_rdiv = torch.mean(rdiv_pred**2)
            loss_ic = torch.mean(
                (u_pred_ics - uv_ics_batch_tens[:, 0:1]) ** 2
                + (v_pred_ics - uv_ics_batch_tens[:, 1:2]) ** 2
            )

            # Normalized losses for BCs
            loss_bc1_u = self.normalize_loss(u_pred_bc1, uv_bc1_batch_tens[:, 0:1])
            loss_bc1_v = self.normalize_loss(v_pred_bc1, uv_bc1_batch_tens[:, 1:2])
            loss_bc2_u = self.normalize_loss(u_pred_bc2, uv_bc2_batch_tens[:, 0:1])
            loss_bc2_v = self.normalize_loss(v_pred_bc2, uv_bc2_batch_tens[:, 1:2])
            loss_bc3_u = self.normalize_loss(u_pred_bc3, uv_bc3_batch_tens[:, 0:1])
            loss_bc3_v = self.normalize_loss(v_pred_bc3, uv_bc3_batch_tens[:, 1:2])
            loss_bc4_u = self.normalize_loss(u_pred_bc4, uv_bc4_batch_tens[:, 0:1])
            loss_bc4_v = self.normalize_loss(v_pred_bc4, uv_bc4_batch_tens[:, 1:2])

            loss_bcs_ic = torch.mean(
                (uv_ics_batch_tens[:, 0:1] - u_pred_ics) ** 2
                + (uv_ics_batch_tens[:, 1:2] - v_pred_ics) ** 2
            )
            loss_bcs = (
                loss_bc1_u
                + loss_bc1_v
                + loss_bc2_u
                + loss_bc2_v
                + loss_bc3_u
                + loss_bc3_v
                + loss_bc4_u
                + loss_bcs_ic
                + +loss_bc4_v
            )

            loss = (
                self.lam_ru_val * loss_ru
                + self.lam_rv_val * loss_rv
                + self.lam_bc_val * loss_bcs
                + self.lam_div_val * loss_rdiv
                + self.lam_ic_val * loss_ic
            )

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            if it % scheduler_step == 0:
                self.my_lr_scheduler.step()

            # Store losses
            if it % 10 == 0:

                self.loss_ic_log.append(loss_ic.detach().cpu().numpy())
                self.loss_bcs_log.append(loss_bcs.detach().cpu().numpy())
                self.loss_ru_log.append(loss_ru.detach().cpu().numpy())
                self.loss_rv_log.append(loss_rv.detach().cpu().numpy())
                self.loss_div_log.append(loss_rdiv.detach().cpu().numpy())
                self.loss_total_log.append(loss.detach().cpu().numpy())

            # Print
            if it % ntk_step == 0:

                print("Epoch:", it, "/", nIter)
                print(
                    "Loss: %.3e, Loss_bcs: %.3e, Loss_ru: %.3e, Loss_rv: %.3e, Loss_rdiv: %.3e, Loss_ic: %.3e"
                    % (loss.item(), loss_bcs, loss_ru, loss_rv, loss_rdiv, loss_ic)
                )
                print(f"lambda_bc: {self.lam_bc_val:3e}")
                print(f"lambda_ru: {self.lam_ru_val:3e}")
                print(f"lambda_rv: {self.lam_rv_val:3e}")
                print(f"lambda_div: {self.lam_div_val:3e}")
                print(f"lambda_ic: {self.lam_ic_val:3e}")
                print("Learning rate: ", self.my_lr_scheduler.get_last_lr()[0])

            if log_NTK:

                if it % ntk_step == 0:
                    print("Compute NTK...")
                    X_bc_batch = np.vstack(
                        [X_bc1_batch, X_bc2_batch, X_bc3_batch, X_bc4_batch]
                    )

                    # Convert to the tensor
                    X_bc_batch_tens = (
                        torch.tensor(X_bc_batch, requires_grad=True).float().to(device)
                    )

                    # Get the parameters of NN
                    params = list(self.nn.parameters())

                    # Store the trace
                    K_bc_value = 0
                    K_ic_value = 0
                    K_ru_value = 0
                    K_rv_value = 0
                    K_div_value = 0

                    u_ic, v_ic, _ = self.net_uvp(
                        X_ics_batch_tens[:, 0:1],
                        X_ics_batch_tens[:, 1:2],
                        X_ics_batch_tens[:, 2:3],
                    )
                    u_bc, v_bc, _ = self.net_uvp(
                        X_bc_batch_tens[:, 0:1],
                        X_bc_batch_tens[:, 1:2],
                        X_bc_batch_tens[:, 2:3],
                    )
                    bc_ntk_pred = torch.cat([u_bc, v_bc], dim=0)

                    res_ntk_u, res_ntk_v, res_ntkdiv = self.net_r(
                        X_res_batch_tens[:, 0:1],
                        X_res_batch_tens[:, 1:2],
                        X_res_batch_tens[:, 2:3],
                    )

                    # Jacobian of the neural networks
                    J_ics = self.compute_jacobian(u_ic, params)
                    J_bc = self.compute_jacobian(bc_ntk_pred, params)
                    J_ru = self.compute_jacobian(res_ntk_u, params)
                    J_rv = self.compute_jacobian(res_ntk_v, params)
                    j_rdiv = self.compute_jacobian(res_ntkdiv, params)

                    # Neural tangent kernels of the neural networks / Trace values
                    K_ic_value = self.compute_ntk(J_ics, self.D2, J_ics, self.D2)
                    K_bc_value = self.compute_ntk(J_bc, self.D1, J_bc, self.D1)
                    K_ru_value = self.compute_ntk(J_ru, self.D3, J_ru, self.D3)
                    K_rv_value = self.compute_ntk(J_rv, self.D3, J_rv, self.D3)
                    K_div_value = self.compute_ntk(j_rdiv, self.D3, j_rdiv, self.D3)

                    # Convert tensor to numpy array
                    K_ic_value = K_ic_value.detach().cpu().numpy()
                    K_bc_value = K_bc_value.detach().cpu().numpy()
                    K_ru_value = K_ru_value.detach().cpu().numpy()
                    K_rv_value = K_rv_value.detach().cpu().numpy()
                    K_div_value = K_div_value.detach().cpu().numpy()

                    trace_K = (
                        np.trace(K_bc_value)
                        + np.trace(K_ru_value)
                        + np.trace(K_rv_value)
                        + np.trace(K_div_value)
                        + np.trace(K_ic_value)
                    )

                    # Store Trace values
                    self.K_bc_log.append(K_bc_value)
                    self.K_ru_log.append(K_ru_value)
                    self.K_rv_log.append(K_rv_value)
                    self.K_div_log.append(K_div_value)
                    self.K_ic_log.append(K_ic_value)

                    if update_lam:
                        self.lam_bc_val = trace_K / np.trace(K_bc_value)
                        self.lam_ru_val = trace_K / np.trace(K_ru_value)
                        self.lam_rv_val = trace_K / np.trace(K_rv_value)
                        self.lam_div_val = trace_K / np.trace(K_div_value)
                        self.lam_ic_val = trace_K / np.trace(K_ic_value)

                        # Store NTK weights
                        self.lam_bc_log.append(self.lam_bc_val)
                        self.lam_ru_log.append(self.lam_ru_val)
                        self.lam_rv_log.append(self.lam_rv_val)
                        self.lam_div_log.append(self.lam_div_val)
                        self.lam_ic_log.append(self.lam_ic_val)

    # Evaluates predictions at test points
    def predict_uvp(self, X_star):

        t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X_star[:, 2:3], requires_grad=True).float().to(device)

        self.nn.eval()

        u_star, v_star, p_star = self.net_uvp(t, x, y)
        u_star = u_star.detach().cpu().numpy()
        v_star = v_star.detach().cpu().numpy()
        p_star = p_star.detach().cpu().numpy()

        return u_star, v_star, p_star

    # Evaluates predictions at test points
    # def predict_r(self, X_star):
    #     X_star = (X_star - self.mu_X) / self.sigma_X

    #     t = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
    #     x = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
    #     y = torch.tensor(X_star[:, 2:3], requires_grad=True).float().to(device)

    #     self.nn.eval()

    #     r_star = self.net_r(t, x)
    #     r_star = r_star.detach().cpu().numpy()
    #     return r_star


def u(x, nu):
    """
    :param x: x = (t, x, y)
    """
    tc = x[:, 0:1]
    xc = x[:, 1:2]
    yc = x[:, 2:3]

    return np.sin(np.pi * xc) * np.cos(np.pi * yc)


def v(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:, 0:1]
    xc = x[:, 1:2]
    yc = x[:, 2:3]

    return np.sin(np.pi * yc) * np.cos(np.pi * xc)


def u_bc(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:, 0:1]
    xc = x[:, 1:2]
    yc = x[:, 2:3]

    return np.sin(np.pi * xc) * np.cos(np.pi * yc) * np.exp(-nu * np.pi**2 * tc)


def v_bc(x, nu):
    """
    :param x: x = (t, x)
    """
    tc = x[:, 0:1]
    xc = x[:, 1:2]
    yc = x[:, 2:3]

    return np.sin(np.pi * yc) * np.cos(np.pi * xc) * np.exp(-nu * np.pi**2 * tc)


def r(u_tt):
    zerores = np.zeros((u_tt.shape[0], 1))
    return zerores


def operator(u, v, p, t, x, y, nu, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0):

    u_t = (
        torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        / sigma_t
    )
    u_x = (
        torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        / sigma_x
    )
    u_y = (
        torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        / sigma_y
    )

    v_t = (
        torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True
        )[0]
        / sigma_t
    )
    v_x = (
        torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True
        )[0]
        / sigma_x
    )
    v_y = (
        torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True
        )[0]
        / sigma_y
    )
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[
        0
    ]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[
        0
    ]

    # Second-order derivatives
    u_xx = (
        torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        / sigma_x
    )
    u_yy = (
        torch.autograd.grad(
            u_y,
            y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True,
        )[0]
        / sigma_y
    )

    v_xx = (
        torch.autograd.grad(
            v_x,
            x,
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        / sigma_x
    )
    v_yy = (
        torch.autograd.grad(
            v_y,
            y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True,
        )[0]
        / sigma_y
    )

    # Residuals
    res_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    res_div = u_x + v_y  # Divergence

    return res_u, res_v, res_div


zero_bc = lambda x: np.zeros((x.shape[0], 2))

res_sampler = Sampler(
    3, dom_coord, lambda x: np.zeros((x.shape[0], 2)), name="Residuals"
)


# PINN model


# t_lb, t_ub = 0, 1
# x_lb, x_ub = 0, 2
# y_lb, y_ub = 0, 2
# coords = np.array([[t_lb, x_lb, y_lb], [t_ub, x_ub, y_ub]])

# # Fix y=2 (index 2), and leave t and x free
# bc_top_sampler = BCSampler(
#     dim=3,
#     coords=coords,
#     func=your_bc_func,  # Function that takes (N,3) array and returns BC values
#     fixed_dims=[2],     # Index of y in [t, x, y]
#     fixed_values=[y_ub]
# )

# x_top, y_top = bc_top_sampler.sample(10)
# print("Sampled BC points:\n", x_top)
# print("BC values:\n", y_top)
print(bc1_coord[0, 1], bc2_coord[0, 1], bc3_coord[0, 2], bc4_coord[0, 2])

# bc1_sampler = BCSampler(
#     3,
#     bc1_coord,
#     get_left_bc_from_history,
#     fixed_dims=[1],
#     fixed_values=[x_lb],
#     name="BC1",
# )
# bc2_sampler = BCSampler(
#     3,
#     bc2_coord,
#     get_right_bc_from_history,
#     fixed_dims=[1],
#     fixed_values=[x_ub],
#     name="BC2",
# )
# bc3_sampler = BCSampler(
#     3,
#     bc3_coord,
#     get_top_bc_from_history,
#     fixed_dims=[2],
#     fixed_values=[yub],
#     name="BC3",
# )
# bc4_sampler = BCSampler(
#     3,
#     bc4_coord,
#     get_bottom_bc_from_history,
#     fixed_dims=[2],
#     fixed_values=[ylb],
#     name="BC4",
# )

# zero_bc = lambda x: np.zeros((x.shape[0], 2))
# bc_cyl_sampler = Sampler()
bc1_sampler = Sampler(
    3, dom_coord, lambda x: np.zeros((x.shape[0], 2)), name="Residuals"
)
bc2_sampler = Sampler(
    3, dom_coord, lambda x: np.zeros((x.shape[0], 2)), name="Residuals"
)
bc3_sampler = Sampler(
    3, dom_coord, lambda x: np.zeros((x.shape[0], 2)), name="Residuals"
)
bc4_sampler = Sampler(
    3, dom_coord, lambda x: np.zeros((x.shape[0], 2)), name="Residuals"
)


# bc1_sampler = Sampler(3, bc1_coord, get_left_bc_from_history, name='BC1')
# bc2_sampler = Sampler(3, bc2_coord, get_right_bc_from_history, name='BC2')
# bc3_sampler = Sampler(3, bc3_coord, get_top_bc_from_history, name='BC3')
# bc4_sampler = Sampler(3, bc4_coord, get_bottom_bc_from_history, name='BC4')
bcs_sampler = [bc1_sampler, bc2_sampler, bc3_sampler, bc4_sampler]
ics_sampler = Sampler(
    3, ics_coord, lambda x: np.hstack([u(x, nu), v(x, nu)]), name="ICs"
)

model = PINN(layers, operator, ics_sampler, bcs_sampler, res_sampler, nu, kernel_size)


def main():
    navier2d.setupNS2d()
    navier2d.mainNS2d()
    train_bool = True
    # train
    if train_bool:

        print("Training the model...")
        model.train(
            iterations, batch_size=kernel_size, log_NTK=log_NTK, update_lam=update_lam
        )

        # save model
        torch.save(model.nn.state_dict(), save_model)
        print(f"Model saved as {save_model}")

        # save losses
        total_loss = []
        total_loss.append(model.loss_total_log)
        total_loss.append(model.loss_ic_log)
        total_loss.append(model.loss_bcs_log)
        total_loss.append(model.loss_ru_log)
        total_loss.append(model.loss_rv_log)
        total_loss.append(model.loss_div_log)
        np.save(save_loss, total_loss)
        print(f"Losses saved as {save_loss}")


def plot_loss():
    losses = np.load(save_loss, allow_pickle=True)
    total_loss = losses[0]
    loss_ic = losses[1]
    loss_bcs = losses[2]
    loss_ru = losses[3]
    loss_rv = losses[4]
    loss_div = losses[5]

    plt.figure(figsize=(10, 6))
    plt.plot(total_loss, label="Total Loss")
    plt.plot(loss_ic, label="Loss ICs")
    plt.plot(loss_bcs, label="Loss BCS")
    plt.plot(loss_ru, label="Loss ru")
    plt.plot(loss_rv, label="Loss rv")
    plt.plot(loss_div, label="Loss div")
    # plt.plot(loss_ic, label='Loss ICS')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.show()


def animate_plot_2d():
    import matplotlib.animation as animation

    # 1) Load model
    model = PINN(
        layers, operator, ics_sampler, bcs_sampler, res_sampler, nu, kernel_size
    )
    model.nn.load_state_dict(torch.load(save_model))

    # 2) Build evaluation grid (2D x-y, loop in time)
    Nx, Ny, Nt = 300, 300, 100
    t_lin = np.linspace(dom_coord[0, 0], dom_coord[1, 0], Nt)
    x_lin = np.linspace(x_lb, x_ub, Nx)
    y_lin = np.linspace(ylb, yub, Ny)
    X, Y = np.meshgrid(x_lin, y_lin)  # both shape (Ny, Nx)

    # 3) Prepare storage for |velocity| frames
    U_frames = []

    with torch.no_grad():
        tot_frame = 0
        for t_val in t_lin:

            if t_val >= 0.0:
                pts = np.column_stack([np.full(X.size, t_val), X.ravel(), Y.ravel()])
                u_pred, v_pred, _ = model.predict_uvp(pts)
                u_pred = u_pred.reshape(Ny, Nx)
                v_pred = v_pred.reshape(Ny, Nx)
                mag = np.sqrt(u_pred**2 + v_pred**2)
                U_frames.append(mag)
                tot_frame += 1

    # 4) Setup the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = np.min(U_frames)
    vmax = np.max(U_frames)
    cax = ax.imshow(
        U_frames[0],
        origin="lower",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        extent=[x_lin.min(), x_lin.max(), y_lin.min(), y_lin.max()],
    )
    title = ax.set_title(f"time = {t_lin[0]:.3f}")
    fig.colorbar(cax, ax=ax, label=r"$|\vec{u}|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 5) Update function
    def update(frame):
        # imshow update: use set_data with a 2D array, not ravel!
        cax.set_data(U_frames[frame])
        title.set_text(f"time = {t_lin[frame]:.3f}")
        return cax, title

    # 6) Animation
    ani = animation.FuncAnimation(
        fig, update, frames=tot_frame, interval=50, blit=False, repeat=True
    )
    plt.show()
    return ani


if __name__ == "__main__":

    def errors():
        import matplotlib.animation as animation

        # 1) Load model
        model = PINN(
            layers, operator, ics_sampler, bcs_sampler, res_sampler, nu, kernel_size
        )
        model.nn.load_state_dict(torch.load(save_model))

        # 2) Build evaluation grid (2D x-y, loop in time)
        Nx, Ny, Nt = 300, 300, 100
        t_lin = np.linspace(dom_coord[0, 0], dom_coord[1, 0], Nt)
        x_lin = np.linspace(x_lb, x_ub, Nx)
        y_lin = np.linspace(ylb, yub, Ny)
        X, Y = np.meshgrid(x_lin, y_lin)  # both shape (Ny, Nx)

        # 3) Prepare storage for |velocity| frames
        U_frames = []

        with torch.no_grad():
            tot_frame = 0
            for t_val in t_lin:

                if t_val <= 0.8:
                    pts = np.column_stack(
                        [np.full(X.size, t_val), X.ravel(), Y.ravel()]
                    )
                    u_pred, v_pred = model.predict_uv(pts)
                    u_pred = u_pred.reshape(Ny, Nx)
                    v_pred = v_pred.reshape(Ny, Nx)
                    mag = np.sqrt(u_pred**2 + v_pred**2)
                    U_frames.append(mag)
                    tot_frame += 1
        # 4) Load CFD data
        dim = 300
        data = np.loadtxt("snapshots.txt", delimiter=",")
        N_space, M_times = data.shape
        print("N_space:", N_space, "M_times:", M_times)
        ux = data[: dim * dim, :]
        uy = data[dim * dim : 2 * dim * dim, :]
        ux_2d = ux.reshape(dim, dim, M_times)
        uy_2d = uy.reshape(dim, dim, M_times)
        umag = np.sqrt(ux_2d**2 + uy_2d**2)
        # Supponiamo:
        # - CFD: umag.shape = (dim, dim, M_times)
        # - PINN: U_frames è una lista di array (Ny, Nx), lunghezza = Nt
        # Bisogna quindi uniformare la shape (Nx, Ny, Nt) per entrambi

        umag = np.array(umag)  # CFD: shape (dim, dim, M_times)
        U_frames = np.array(U_frames)  # PINN: shape (Nt, Ny, Nx)
        U_frames = np.transpose(U_frames, (1, 2, 0))  # now (Ny, Nx, Nt) se necessario
        # Controlla dimensioni!
        print("CFD shape:", umag.shape, "PINN shape:", U_frames.shape)

        # (Se necessario, ridimensiona/interpola uno dei due dataset!)

        # Calcolo errore L2 frame by frame
        errors = []
        for t in range(min(U_frames.shape[2], umag.shape[2])):
            u_pinn = U_frames[..., t]
            u_cfd = umag[..., t]
            err = np.sqrt(np.mean((u_pinn - u_cfd) ** 2))  # L2 globale
            errors.append(err)

        # Plot errore temporale
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(errors)
        plt.xlabel("Time frame")
        plt.ylabel("RMS Error")
        plt.title("RMS Error PINN vs CFD")
        plt.grid(True)
        plt.show()

        def animation_err():
            # Assumiamo che le griglie coincidano (stesse Nx, Ny, Nt, dim)
            diff_frames = []
            for t in range(min(U_frames.shape[2], umag.shape[2])):
                u_pinn = U_frames[..., t]
                u_cfd = umag[..., t]
                diff = u_pinn - u_cfd
                diff_frames.append(diff)
            diff_frames = np.array(diff_frames)  # shape (Nt, Ny, Nx) o (Nt, dim, dim)

            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(6, 5))
            vmax = np.max(np.abs(diff_frames))
            im = ax.imshow(
                diff_frames[0], origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax
            )
            ax.set_title(f"Difference (PINN - CFD), frame 0")
            fig.colorbar(im, ax=ax, label="Difference")

            def update(frame):
                im.set_data(diff_frames[frame])
                ax.set_title(f"Difference (PINN - CFD), frame {frame}")
                return [im]

            ani = animation.FuncAnimation(
                fig, update, frames=diff_frames.shape[0], interval=50, blit=True
            )
            plt.show()

        animation_err()

    # print bcs
    # print(bc1_sampler)
    # load the model
    # burgers2d.setupB2d()  # Initialize the CFD simulation
    # burgers2d.mainB2d()
    main()
    # errors()
    # plot_loss()
    # plot()
    animate_plot_2d()
    pass
# plot analytical initial conditions
