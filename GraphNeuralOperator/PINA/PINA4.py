import torch
import matplotlib.pyplot as plt
import warnings

from pina import Trainer
from pina.solver import SupervisedSolver
from pina.model import KernelNeuralOperator
from pina.model.block import FourierBlock1D
from pina.problem.zoo import SupervisedProblem

"""Neural Operators (NOs) are a class of machine learning models designed 
to learn mappings between function spaces, unlike traditional neural networks 
which learn mappings between finite-dimensional vectors. In the context of 
differential equations, this means a Neural Operator can learn the solution operator.
1. mesh free learning
2. Fast inference
3. Physics-aware extensions"""


# 1D Advection Equation Experiment
def generate_data(n_samples, x, c=1, t=0.5):
    x = x.T.repeat(n_samples, 1)
    u0 = torch.zeros_like(x)
    ut = torch.zeros_like(x)
    for k in range(1, 4):
        amplitude = torch.rand(n_samples, 1) * 0.5
        phase = torch.rand(n_samples, 1) * 2 * torch.pi
        u0 += amplitude * torch.sin(2 * torch.pi * k * x + phase)
        shif_x = (x - c * t) % 1.0
        ut += amplitude * torch.sin(2 * torch.pi * k * shif_x + phase)
    return u0, ut


x_train = torch.linspace(0, 1, 100).reshape(-1, 1)
input, target = generate_data(1000, x_train)


class encoder(torch.nn.Module):
    """simple linear layer mapping the input dimension to the hidden dimension"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.enc = torch.nn.Linear(1, hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.enc(x)
        return x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)


class decoder(torch.nn.Module):
    """two linear layers mapping the hidden dimension to 128 and back to the input dimension"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dec(x)
        return x.squeeze(-1)  # (batch_size, seq_len)


class Processor(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.proc = torch.nn.Sequential(
            FourierBlock1D(64, 64, 8, torch.nn.ReLU),
            FourierBlock1D(64, 64, 8, torch.nn.ReLU),
        )

    def forward(self, x):
        return self.proc(x)


# define the model
model = KernelNeuralOperator(
    lifting_operator=encoder(),
    integral_kernels=Processor(),
    projection_operator=decoder(),
)

# define the problem
problem = SupervisedProblem(input, target)

# define the solver
solver = SupervisedSolver(problem, model, use_lt=False)

# train the model
trainer = Trainer(
    solver,
    max_epochs=10,
    train_size=0.8,
    test_size=0.2,
    batch_size=256,
    accelerator="gpu",
    enable_model_summary=False,
)

trainer.train()
_ = trainer.test()

torch.save(model.state_dict(), "neural_operator.pth")
print("Model saved as neural_operator.pth")


def plot_solution():
    input, target = generate_data(1000, x_train)
    prediction = solver(input).detach()

    # plot
    plt.plot(x_train, input[0], label=f"Input u(x, t=0)")
    plt.plot(x_train, target[0], label=f"Target u(x, t=0.5)")
    plt.plot(x_train, prediction[0], "--r", label=f"NO prediction u(x, t=0.5)")
    plt.title("Generated 1D Advection Data")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_solution()
