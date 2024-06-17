import numpy as np

def copy_to_final(curr_path, final_path, n):
	final_path[:n + 1] = curr_path[:]
	final_path[n] = curr_path[0]

def first_min(adj, i):
    return np.min(adj[i, :])

def second_min(adj, i):
    second = np.min(adj[adj!=first_min(adj, i)])
    return second

class BnB_TSP:
    def __init__(self, adj) -> None:
        N = adj.shape[0]
        self.adj = adj.copy()
        self.visited = [False] * N
        self.visited[0] = True
        
        self.curr_path = [-1] * (N + 1)
        self.curr_path[0] = 0
        
        self.curr_bound = 0
        self.final_path = [None] * (N + 1)
        self.total_cost = np.inf
        
        for i in range(N):
            self.curr_bound += (first_min(adj, i) + second_min(adj, i))

        self.curr_bound = np.ceil(self.curr_bound / 2)
        
    def _solve(self):
        self._tsp_recursive(0, 1)
        
    def _tsp_recursive(self, curr_weight, level):
        n = self.adj.shape[0]
        if level == n:
            if self.adj[self.curr_path[level - 1]][self.curr_path[0]] != 0:
                curr_res = curr_weight + self.adj[self.curr_path[level - 1]][self.curr_path[0]]
                if curr_res < self.total_cost:
                    copy_to_final(self.curr_path, self.final_path, n)
                    self.total_cost = curr_res
            return

        for i in range(n):
            if (self.adj[self.curr_path[level-1]][i] != 0 and
                self.visited[i] == False):
                temp = self.curr_bound
                curr_weight += self.adj[self.curr_path[level - 1]][i]

                if level == 1:
                    self.curr_bound -= ((first_min(self.adj, self.curr_path[level - 1]) + first_min(self.adj, i)) / 2)
                else:
                    self.curr_bound -= ((second_min(self.adj, self.curr_path[level - 1]) + first_min(self.adj, i)) / 2)

                if self.curr_bound + curr_weight < self.total_cost:
                    self.curr_path[level] = i
                    self.visited[i] = True
                    self._tsp_recursive(curr_weight, level + 1)

                curr_weight -= self.adj[self.curr_path[level - 1]][i]
                self.curr_bound = temp

                self.visited = [False] * len(self.visited)
                for j in range(level):
                    if self.curr_path[j] != -1:
                        self.visited[self.curr_path[j]] = True
    
    @property
    def is_solved(self):
        return all(self.visited)
    
    @staticmethod
    def solve(adj):
        problem = BnB_TSP(adj)
        problem._solve()
        return problem.final_path[:-1], problem.total_cost
    
    

