import torch
import matplotlib.pyplot as plt

# neural operator tutorial number 21

from pina import Trainer
from pina.solver import SupervisedSolver
from pina.model import KernelNeuralOperator
from pina.model.block import FourierBlock1D
from pina.problem.zoo import SupervisedProblem


def generate_data(n_samples, x, c=1, t=0.5):
    x = x.T.repeat(n_samples, 1)
    u0 = torch.zeros_like(x)
    ut = torch.zeros_like(x)
    for k in range(1, 4):
        A = torch.rand(n_samples, 1) * 0.5
        phi = torch.rand(n_samples, 1) * 2 * torch.pi
        u0 += A * torch.sin(2 * torch.pi * k * x + phi)
        shift_x = (x - c * t) % 1.0  # periodic shift
        ut += A * torch.sin(2 * torch.pi * k * shift_x + phi)
    return u0, ut


x_train = torch.linspace(0, 1, 100).reshape(-1, 1)

input, target = generate_data(1000, x_train)


def visualize_data(input, target):
    plt.plot(x_train, input[0], label=f"Input u(x, t=0)")
    plt.plot(x_train, target[0], label=f"Target u(x, t=0.5)")
    plt.title("Generated 1D Advection Data")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.show()


# encoder simple linear layer 1->64
class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.enc = torch.nn.Linear(1, hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)  # ensure input is 2D [B, Nx] -> [B, Nx, 1]
        x = self.enc(x)  # [B, Nx, 1] -> [B, Nx, hidden_dim]
        return x.permute(0, 2, 1)  # [B, hidden_dim, Nx]


# decoder two linear layers 64->128->1
class Decoder(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, hidden_dim, Nx] -> [B, Nx, hidden_dim]
        x = self.dec(x)
        return x.squeeze(-1)  # [B, Nx, 1] -> [B, Nx]


# processor: two FNO blocks of size 64
class Processor(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.proc = torch.nn.Sequential(
            FourierBlock1D(64, 64, 8, torch.nn.ReLU),
            FourierBlock1D(64, 64, 8, torch.nn.ReLU),
        )

    def forward(self, x):
        return self.proc(x)


# model
model = KernelNeuralOperator(
    lifting_operator=Encoder(),
    integral_kernels=Processor(),
    projection_operator=Decoder(),
)

# making the problem
problem = SupervisedProblem(input, target)
solver = SupervisedSolver(problem, model, use_lt=False)

# training
trainer = Trainer(
    solver,
    max_epochs=100,
    train_size=0.8,
    test_size=0.2,
    batch_size=256,
    accelerator="gpu",
    enable_model_summary=False,
)

trainer.train()
_ = trainer.test()


def visualize_prediction():
    input, target = generate_data(100, x_train)
    pred = solver(input).detach()
    plt.plot(x_train, input[0], label=f"Input u(x, t=0)")
    plt.plot(x_train, target[0], label=f"Target u(x, t=0.5)")
    plt.plot(x_train, pred[0], "--r", label=f"NO prediction u(x, t=0.5)")
    plt.title("Generated 1D Advection Data")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.show()


visualize_prediction()
