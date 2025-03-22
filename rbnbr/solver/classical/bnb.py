from rbnbr.solver.solver_base import SolverBase
from rbnbr.problems.problem_base import ProblemBase


class BaseBnBMaxCutSolver(SolverBase):
    def __init__(self, problem: ProblemBase):
        super().__init__(problem)

    def solve(self):
        pass


    def search_heuristic(self):
        pass 