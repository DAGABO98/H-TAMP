"Code based on: https://github.com/vislearn/FrEIA/blob/master/FrEIA/modules/splines/rational_quadratic.py"

from typing import Tuple

import torch

def rational_quadratic_spline(x: torch.Tensor,
                              left: torch.Tensor,
                              right: torch.Tensor,
                              bottom: torch.Tensor,
                              top: torch.Tensor,
                              deltas_left: torch.Tensor,
                              deltas_right: torch.Tensor,
                              rev: bool = False,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rational-quadratic spline with the algorithm described in arXiv:1906.04032
    Forward output is defined for each bin by the fraction
    ..math::
        z = f(x) = \\frac{ \\beta_0 + \\beta_1 x + \\beta_2 x^2 }{ 1 + \\beta_3 x + \\beta_4 x^2 }

    where the \\beta are constrained to yield a smooth function over all bins
    Args:
        x: input tensor
        left: left bin edges for inputs inside the spline domain
        right: same as left
        bottom: same as left
        top: same as left
        deltas_left: spline derivatives at the left and bottom bin edge
        deltas_right: spline derivatives at the right and top bin edge
        rev: whether to run in reverse

    Returns:
        splined output tensor
        log jacobian matrix entries
    """
    # rename variables to match the paper:
    # xk means $x_k$ and xkp means $x_{k+1}$
    xk = left
    xkp = right
    yk = bottom
    ykp = top
    dk = deltas_left
    dkp = deltas_right

    # define some commonly used values
    dx = xkp - xk
    dy = ykp - yk
    sk = dy / dx

    if not rev:
        xi = (x - xk) / dx

        # Eq 4 in the paper
        numerator = dy * (sk * xi ** 2 + dk * xi * (1 - xi))
        denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
        out = yk + numerator / denominator
    else:
        y = x
        # Eq 6-8 in the paper
        a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
        b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
        c = -sk * (y - yk)

        # Eq 29 in the appendix of the paper
        discriminant = b ** 2 - 4 * a * c
        if not torch.all(discriminant >= 0):
            raise(RuntimeError(f"Discriminant must be positive, but is violated by {torch.min(discriminant)}"))

        xi = 2 * c / (-b - torch.sqrt(discriminant))

        out = xi * dx + xk

    # Eq 5 in the paper
    numerator = sk ** 2 * (dkp * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
    log_jac = torch.log(numerator) - torch.log(denominator)

    return out, log_jac