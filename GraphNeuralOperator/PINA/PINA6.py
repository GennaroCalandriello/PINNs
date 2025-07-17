import torch
import matplotlib.pyplot as plt

# multiscale problem

from pina import Condition, Trainer
from pina.problem import SpatialProblem
from pina.operator import laplacian
from pina.solver import PINN, SelfAdaptivePINN
from pina.loss import LpLoss
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.model import FeedForward
from pina.model.block import FourierFeatureEmbedding


class Poisson(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})

    def poisson_equation(input_, output_):
        x = input_.extract("x")
        u_xx = laplacian(output_, input_, components=["u"], d=["x"])
        f = ((2 * torch.pi) ** 2) * torch.sin(2 * torch.pi * x) + 0.1 * (
            (50 * torch.pi) ** 2
        ) * torch.sin(50 * torch.pi * x)
        return u_xx + f

    domains = {
        "bc0": CartesianDomain({"x": 0.0}),
        "bc1": CartesianDomain({"x": 1.0}),
        "phys": spatial_domain,
    }

    conditions = {
        "bc0": Condition(domain="bc0", equation=FixedValue(0.0)),
        "bc1": Condition(domain="bc1", equation=FixedValue(0.0)),
        "phys": Condition(domain="phys", equation=Equation(poisson_equation)),
    }

    def solution(self, x):
        return torch.sin(2 * torch.pi * x) + 0.1 * torch.sin(50 * torch.pi * x)


problem = Poisson()
# discretization
problem.discretise_domain(128, "grid", domains=["phys"])
problem.discretise_domain(1, "grid", domains=["bc0", "bc1"])


def all(version=2):
    """A simple FeedForward network struggles to handle multiscale problems,
    especially when there are not enough collocation points to capture the
    different scales effectively."""

    if version == 1:

        # train with PINN
        pinn = PINN(
            problem=problem,
            model=FeedForward(
                input_dimensions=1,
                output_dimensions=1,
                layers=[100, 100, 100],
            ),
        )
        trainer = Trainer(
            pinn,
            max_epochs=1000,
            accelerator="gpu",
            enable_model_summary=False,
            val_size=0.0,
            train_size=1.0,
            test_size=0.0,
        )
        trainer.train()

    if version == 2:
        pinn = SelfAdaptivePINN(
            problem=problem,
            model=FeedForward(
                input_dimensions=1,
                output_dimensions=1,
                layers=[100, 100, 100],
            ),
        )
        trainer = Trainer(
            pinn,
            max_epochs=1000,
            accelerator="gpu",
            enable_model_summary=False,
            val_size=0.0,
            train_size=1.0,
            test_size=0.0,
        )
        trainer.train()
    return pinn


class MultiscaleFourierNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(
            input_dimension=1, output_dimension=100, sigma=1
        )
        self.embedding2 = FourierFeatureEmbedding(
            input_dimension=1, output_dimension=100, sigma=10
        )
        self.layers = FeedForward(
            input_dimensions=100, output_dimensions=100, layers=[100]
        )
        self.final_layer = torch.nn.Linear(2 * 100, 1)

    def forward(self, x):
        x1 = self.layers(self.embedding1(x))
        x2 = self.layers(self.embedding2(x))
        return self.final_layer(torch.cat([x1, x2], dim=-1))


def all2():
    multiscale_pinn = PINN(problem=problem, model=MultiscaleFourierNet())
    trainer = Trainer(
        multiscale_pinn,
        max_epochs=1000,
        accelerator="gpu",
        enable_model_summary=False,
        val_size=0.0,
        train_size=1.0,
        test_size=0.0,
    )
    trainer.train()
    return multiscale_pinn


def plot_results(pinn_used, title):
    pts = pinn_used.problem.spatial_domain.sample(256, "grid", variables="x")
    pred_output = pinn_used(pts).extract("u").tensor.detach()
    true_output = pinn_used.problem.solution(pts).detach()
    plt.plot(pts.extract(["x"]), pred_output, label="NN prediction")
    plt.plot(pts.extract(["x"]), true_output, label="True solution")
    plt.title(title)
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    pinn = all2()
    plot_results(pinn, "PINN Solution to Poisson Equation")
    # plot_results(pinn, "PINN Solution to Poisson Equation")
