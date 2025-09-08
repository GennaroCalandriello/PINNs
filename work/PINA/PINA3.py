import torch
import matplotlib.pyplot as plt
import warnings
from pina import Condition, LabelTensor, Trainer
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import laplacian, grad
from pina.domain import CartesianDomain
from pina.solver import PINN
from pina.equation import Equation, FixedValue
from pina.callback import MetricTracker
import matplotlib.animation as animation

# hyperpar
max_epochs = 1000


def wave_equation_2D(input_, output_):
    u_t = grad(output_, input_, components=["u"], d=["t"])
    u_tt = grad(u_t, input_, components=["dudt"], d=["t"])
    nabla_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return nabla_u - u_tt


def initial_conditions(input_, output_):
    u_expected = torch.sin(torch.pi * input_.extract(["x"])) * torch.sin(
        torch.pi * input_.extract(["y"])
    )
    return output_.extract(["u"]) - u_expected


class Wave2D(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})
    domains = {
        "bcx_up": CartesianDomain({"x": 1, "y": [0, 1], "t": [0, 1]}),
        "bcx_down": CartesianDomain({"x": 0, "y": [0, 1], "t": [0, 1]}),
        "bcy_up": CartesianDomain({"x": [0, 1], "y": 1, "t": [0, 1]}),
        "bcy_down": CartesianDomain({"x": [0, 1], "y": 0, "t": [0, 1]}),
        "initial": CartesianDomain({"x": [0, 1], "y": [0, 1], "t": 0}),
        "D": CartesianDomain({"x": [0, 1], "y": [0, 1], "t": [0, 1]}),
    }
    conditions = {
        "bcx_up": Condition(domain="bcx_up", equation=FixedValue(0.0)),
        "bcx_down": Condition(domain="bcx_down", equation=FixedValue(0.0)),
        "bcy_up": Condition(domain="bcy_up", equation=FixedValue(0.0)),
        "bcy_down": Condition(domain="bcy_down", equation=FixedValue(0.0)),
        "initial": Condition(domain="initial", equation=Equation(initial_conditions)),
        "D": Condition(domain="D", equation=Equation(wave_equation_2D)),
    }

    def solution(self, pts):
        f = (
            torch.sin(torch.pi * pts.extract(["x"]))
            * torch.sin(torch.pi * pts.extract(["y"]))
            * torch.cos(torch.sqrt(torch.tensor(2.0)) * torch.pi * pts.extract(["t"]))
        )
        return LabelTensor(f, self.output_variables)


# Imposing Hard Constraints: u_pinn = xy(1-x)(1-y)*NN(x, y, t) ---> da capire meglio


class HardConstraintMLP(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, output_dim),
        )

    def forward(self, x):
        hard = (
            x.extract(["x"])
            * x.extract(["y"])
            * (1 - x.extract(["x"]))
            * (1 - x.extract(["y"]))
        )
        return hard * self.layers(x)


# siccome la soluzione fa schifo, imposto un hard constraint anche su t:
# u_pinn = xy(1-x)(1-y)*NN(x, y, t)*t +cos(sqrt(2)*pi*t)*sin(pi*x)*sin(pi*y)
class HardConstraintMLPTime(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, output_dim),
        )

    def forward(self, x):
        hard_s = (
            x.extract(["x"])
            * x.extract(["y"])
            * (1 - x.extract(["x"]))
            * (1 - x.extract(["y"]))
        )
        hard_t = (
            torch.sin(torch.pi * x.extract(["x"]))
            * torch.sin(torch.pi * x.extract(["y"]))
            * torch.cos(torch.sqrt(torch.tensor(2.0)) * torch.pi * x.extract(["t"]))
        )
        return hard_s * self.layers(x) * x.extract(["t"]) + hard_t


def train():
    # initialize the problem
    problem = Wave2D()
    # data generation
    problem.discretise_domain(1000, "random", domains="all")

    # define the model (time constraint)
    model = HardConstraintMLPTime(
        len(problem.input_variables), len(problem.output_variables)
    )

    # create the solver
    pinn = PINN(problem=problem, model=model)

    # create the trainer
    trainer = Trainer(
        solver=pinn,
        max_epochs=max_epochs,
        accelerator="gpu",
        enable_model_summary=False,
        train_size=1.0,
        val_size=0.0,
        test_size=0.0,
        callbacks=[MetricTracker(["train_loss", "initial_loss", "D_loss"])],
    )

    # train the MODEL
    trainer.train()
    trainer_metrics = trainer.callbacks[0].metrics
    # save the model
    torch.save(pinn.state_dict(), "wave2d.pth")
    print("Model saved as wave2d.pth")

    # for metric, loss in trainer_metrics.items():
    #     plt.plot(range(len(loss)), loss, label=metric)

    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.yscale("log")
    # plt.title("Training Losses")
    # plt.legend()
    # plt.show()


@torch.no_grad()
def plot_solution(time):
    problem = Wave2D()
    model = HardConstraintMLPTime(
        len(problem.input_variables), len(problem.output_variables)
    )
    solver = PINN(problem=problem, model=model)
    solver.load_state_dict(torch.load("wave2d.pth"))
    problem = solver.problem
    spatial_samples = problem.spatial_domain.sample(100, "grid")
    time = LabelTensor(torch.tensor([[time]]), "t")
    # cross data
    points = spatial_samples.append(time, mode="cross")

    data = {
        "PINN solution": solver(points),
        "True solution": problem.solution(points),
        "Absolute difference": torch.abs(solver(points) - problem.solution(points)),
    }
    plt.suptitle(f"Solution at t = {time.item()}")
    for idx, (title, field) in enumerate(data.items()):
        plt.subplot(1, 3, idx + 1)
        plt.title(title)
        plt.tricontourf(
            points.extract("x").tensor.flatten(),
            points.extract("y").tensor.flatten(),
            field.tensor.flatten(),
        )
        plt.colorbar()
        plt.tight_layout()
    plt.show()


@torch.no_grad()
def animate_solution(t_start=0.0, t_end=1.0, n_frames=50, interval=100):
    problem = Wave2D()
    model = HardConstraintMLPTime(
        len(problem.input_variables), len(problem.output_variables)
    )
    solver = PINN(problem=problem, model=model)
    solver.load_state_dict(torch.load("wave2d.pth"))
    problem = solver.problem
    spatial_samples = problem.spatial_domain.sample(100, "grid")

    x = spatial_samples.extract("x").tensor.flatten()
    y = spatial_samples.extract("y").tensor.flatten()

    times = torch.linspace(t_start, t_end, n_frames)

    ti = times[0].item()
    time = LabelTensor(torch.tensor([[ti]]), "t")
    points = spatial_samples.append(time, mode="cross")
    pinn_sol = solver(points)
    true_sol = problem.solution(points)
    abs_diff = torch.abs(pinn_sol - true_sol)
    data = [
        pinn_sol.tensor.flatten(),
        true_sol.tensor.flatten(),
        abs_diff.tensor.flatten(),
    ]
    # Create the figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # add colorbars
    for ax in axs:
        cf = ax.tricontourf(
            x,
            y,
            data[2],
            levels=30,
        )
        fig.colorbar(cf, ax=ax)

    plt.subplots_adjust(top=0.85)
    fig.suptitle("Soluzione 2D Wave Equation - PINN vs Analitica")

    titles = ["PINN solution", "True solution", "Absolute difference"]
    contourfs = [None, None, None]  # Solo per compatibilità

    def update(frame):
        ti = times[frame].item()
        time = LabelTensor(torch.tensor([[ti]]), "t")
        points = spatial_samples.append(time, mode="cross")
        pinn_sol = solver(points)
        true_sol = problem.solution(points)
        abs_diff = torch.abs(pinn_sol - true_sol)
        data = [
            pinn_sol.tensor.flatten(),
            true_sol.tensor.flatten(),
            abs_diff.tensor.flatten(),
        ]

        for i, (field, ax) in enumerate(zip(data, axs)):
            ax.clear()  # Questo elimina TUTTE le collezioni precedenti
            cf = ax.tricontourf(x, y, field, levels=30)
            ax.set_title(titles[i])

        fig.suptitle(f"Soluzioni 2D, t = {ti:.3f}")

        # Attenzione: non serve restituire nulla con blit=False
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=False
    )

    plt.show()


if __name__ == "__main__":
    # plot_solution(0.2)
    animate_solution(t_start=0.0, t_end=1.0, n_frames=60, interval=100)
