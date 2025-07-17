# Burgers Equation
# =================
from pina import Condition, Trainer, LabelTensor
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.equation import Equation, FixedValue
from pina.operator import grad, fast_grad, laplacian
from pina.domain import CartesianDomain
from pina.model import FeedForward
from pina.solver import PINN
from pina.optim import TorchOptimizer
import matplotlib.pyplot as plt

import torch


# capire quaaa!!!
# c'è una dinamica simbolica??? Penso di sì
def hamburgers1D(input_, output_):
    du = fast_grad(output_, input_, components=["u"], d=["t"])
    dux = grad(output_, input_, components=["u"], d=["x"])
    ddux = fast_grad(dux, input_, components=["dudx"], d=["x"])
    return (
        du.extract(["dudt"])
        + output_.extract(["u"]) * dux.extract(["dudx"])
        - (0.01 / torch.pi) * ddux.extract(["ddudxdx"])
    )


def initial_conditions(input_, output_):
    u_expected = -torch.sin(torch.pi * input_.extract(["x"]))
    return output_.extract(["u"]) - u_expected


class Burgers1D(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-10, 10]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "bc1": CartesianDomain({"x": -1, "t": [0, 1]}),
        "bc2": CartesianDomain({"x": 1, "t": [0, 1]}),
        "time_cond": CartesianDomain({"x": [-1, 1], "t": 0}),
        "phys_cond": CartesianDomain({"x": [-1, 1], "t": [0, 5]}),
    }

    conditions = {
        "bc1": Condition(domain="bc1", equation=FixedValue(0.0)),
        "bc2": Condition(domain="bc2", equation=FixedValue(0.0)),
        "time_cond": Condition(
            domain="time_cond", equation=Equation(initial_conditions)
        ),
        "phys_cond": Condition(domain="phys_cond", equation=Equation(hamburgers1D)),
    }


# sampling for training
problem = Burgers1D()
problem.discretise_domain(n=1000, mode="random", domains=["time_cond"])
problem.discretise_domain(n=1000, mode="random", domains=["phys_cond"])
problem.discretise_domain(n=10, mode="random", domains=["bc1"])
problem.discretise_domain(n=10, mode="random", domains=["bc2"])

# build the model
model = FeedForward(
    layers=[10, 10], func=torch.nn.Tanh, output_dimensions=1, input_dimensions=2
)

pinn = PINN(problem, model, TorchOptimizer(torch.optim.RAdam, lr=0.001))

# creating the trainer
trainer = Trainer(
    solver=pinn,
    max_epochs=200,
    accelerator="gpu",
)

# train
trainer.train()
torch.save(pinn.state_dict(), "burgers1d.pth")
print("Model saved as burgers1d.pth")


@torch.no_grad()
def plot_solution(solver, time):
    # get the problem
    problem = solver.problem
    # get spatial points
    spatial_samples = problem.spatial_domain.sample(100, "grid")
    # get temporal value
    time = LabelTensor(torch.tensor([[time]]), "t")
    # cross data
    points = spatial_samples.append(time, mode="cross")
    # compute pinn solution, true solution and absolute difference
    data = {
        "PINN solution": solver(points),
    }
    # plot the solution

    plt.figure(figsize=(8, 4))
    plt.plot(
        points.extract("x"),
        data["PINN solution"].extract("u").tensor.detach(),
        label="PINN solution",
        color="blue",
    )
    plt.title(f"Solution at t={time.extract('t').item()}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid()
    plt.show()


# # prediction and plotting
pts = pinn.problem.spatial_domain.sample(256, "grid", variables="x")
# pred = pinn.forward(pts).extract("u").tensor.detach()
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# ax.plot(pts.extract("x"), pred, label="Predicted")
# plt.show()
if __name__ == "__main__":
    plot_solution(pinn, 2.9)
    plt.show()
