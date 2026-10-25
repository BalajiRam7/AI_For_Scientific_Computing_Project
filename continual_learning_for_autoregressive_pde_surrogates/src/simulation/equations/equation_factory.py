from src.simulation.equations.advdiff1d import AdvectionDiffusionEquation1D
from src.simulation.equations.advdiff2d import AdvectionDiffusionEquation2D
from src.simulation.equations.burgers1d import BurgersEquation1D
from src.simulation.equations.burgers2d import BurgersEquation2D
from src.simulation.equations.navierstokes2d import NavierStokes2D


def create_equation(
    equation_name,
    params,
):
    if equation_name == "advdiff1d":
        return AdvectionDiffusionEquation1D(**params)
    elif equation_name == "advdiff2d":
        return AdvectionDiffusionEquation2D(**params)
    elif equation_name == "burgers1d":
        return BurgersEquation1D(**params)
    elif equation_name == "burgers2d":
        return BurgersEquation2D(**params)
    elif equation_name == "navierstokes2d":
        return NavierStokes2D(**params)
    else:
        raise NotImplementedError(f"Unknown equation: {equation_name}")
