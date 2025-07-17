# Studio dei domini in PINA
# =========================
import torch
import matplotlib.pyplot as plt
from pina import LabelTensor
from pina.domain import (
    EllipsoidDomain,
    Difference,
    CartesianDomain,
    Union,
    SimplexDomain,
    DomainInterface,
)


def plott(ax, pts, title):

    ax.title.set_text(title)
    ax.scatter(pts.extract("x"), pts.extract("y"), color="red", alpha=0.5)

    # create some domains


cartesian = CartesianDomain({"x": [0, 1], "y": [0, 2]})
ellipsoid_no_border = EllipsoidDomain({"x": [0, 0.5], "y": [0, 0.5]})
ellipsoid_with_border = EllipsoidDomain({"x": [0, 2], "y": [1, 2]}, sample_surface=True)

# now we must sample points on this domains:
cartesian_sample = cartesian.sample(n=1000, mode="random")
ellipsoid_no_border_sample = ellipsoid_no_border.sample(n=1000, mode="random")
ellipsoid_with_border_sample = ellipsoid_with_border.sample(n=1000, mode="random")

# Simplices domains
simplexDomain1 = SimplexDomain(
    [
        LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[1, 0]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
    ]
)

simplexDomain2 = SimplexDomain(
    [
        LabelTensor(torch.tensor([[0.0, -2.0]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[0.5, 0.5]]), labels=["x", "y"]),
        LabelTensor(torch.tensor([[-2.0, 0.0]]), labels=["x", "y"]),
    ]
)


def sample_domains():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    pts_list = [
        cartesian_sample,
        ellipsoid_no_border_sample,
        ellipsoid_with_border_sample,
    ]
    titoli = [
        "Cartesian Domain",
        "Ellipsoid Domain (no border)",
        "Ellipsoid Domain (with border)",
    ]
    for i, pts in enumerate(pts_list):
        plott(ax[i], pts, titoli[i])


def simplex_domains():
    # ora proviamo simplessi:
    # sampling
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    pts2 = simplexDomain2.sample(n=10000, mode="random")
    plott(axs, pts2, "Simplex Domain 2")


# boolean operations on domains
def booleanOperations():
    CartesianEllipseNoBOrderUnion = Union([cartesian, ellipsoid_no_border])
    CartesianEllipseBorderUnion = Union([cartesian, ellipsoid_with_border])
    three_doms = Union([cartesian, ellipsoid_no_border, ellipsoid_with_border])
    simplex_union = Union([simplexDomain1, simplexDomain2])
    simplex_ellipse_union = Union([simplexDomain1, ellipsoid_no_border])
    simplexCartesianDiff = Difference(
        [simplexDomain1, ellipsoid_no_border]
    )  # this is a difference operation

    # samplings
    CE_noborder_sample = CartesianEllipseNoBOrderUnion.sample(n=1000, mode="random")
    CE_border_sample = CartesianEllipseBorderUnion.sample(n=1000, mode="random")
    three_doms_sample = three_doms.sample(n=1000, mode="random")
    simplex_union_sample = simplex_union.sample(n=1000, mode="random")
    simplex_ellipse_union_sample = simplex_ellipse_union.sample(n=1000, mode="random")
    simplex_cartesian_diff_sample = simplexCartesianDiff.sample(n=1000, mode="random")

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    pts_list = [
        simplex_union_sample,
        simplex_cartesian_diff_sample,
    ]
    titoli = [
        "Cartesian + Ellipsoid (no border)",
        "Cartesian + Ellipsoid (with border)",
        "Cartesian + Ellipsoid (no border) + Ellipsoid (with border)",
        "Simplex Domain 1 + Simplex Domain 2",
        "Simplex Domain 1 + Ellipsoid (no border)",
        "Simplex Domain 1 - Cartesian Domain",
    ]

    for i, pts in enumerate(pts_list):
        plott(ax[i], pts, "fig")
    plt.show()


if __name__ == "__main__":
    # sample_domains()
    # simplex_domains()
    booleanOperations()
