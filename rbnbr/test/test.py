import numpy as np
from rbnbr.classical.gw import solve_sdp


def test_solve_sdp():
    W = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
    X = solve_sdp(W)
    print(X)

