import numpy as np
from scipy import spatial
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Test values
# num_points = 45 # number of peaks
# s_pop = 40  # number of ants
# m_iter = 10

num_points = int(sys.argv[1]) # number of peaks
s_pop = int(sys.argv[2])  # number of ants
m_iter = int(sys.argv[3]) # maximum iterations
_alpha = int(sys.argv[4]) # maximum iterations
_beta = int(sys.argv[5]) # maximum iterations
_rho = float(sys.argv[6]) # maximum iterations


points_coordinate = np.random.rand(num_points, 2)  # generate random peaks
print("Coordinates of peaks:\n", points_coordinate[:10], "\n")

#calculation of the matrix of distances between vertices, calculul matrixei distanțelor dintre vârfuri
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
print("Matrix distance:\n", distance_matrix)
class ACO_TSP:  # class of ant colony algorithm for solving the traveling salesman problem
    def __init__(self, func, n_dim, size_pop=10, max_iter=20, distance_matrix=None, alpha=1, beta=2, rho=0.1):
        self.func = func
        self.n_dim = n_dim  #number of cities
        self.size_pop = size_pop  # number of ants
        self.max_iter = max_iter  #number of iterations
        self.alpha = alpha  # coefficient of the importance of pheromones in choosing a path
        self.beta = beta  # distance significance factor
        self.rho = rho  # pheromone evaporation rate,

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))

        # Pheromone matrix updated every iteration,
        self.Tau = np.ones((n_dim, n_dim))
        # The path of each ant in a certain generation,
        self.Table = np.zeros((size_pop, n_dim)).astype(int)
        self.y = None  #The total distance traveled by an ant in a given generation,
        self.generation_best_X, self.generation_best_Y = [], [] # fixing the best generations,
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y
        self.best_x, self.best_y = None, None

    def run(self, max_iter=None): 
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            # transition probability without normalization
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta
            for j in range(self.size_pop):  # for each ant
                # starting point of the path (it can be random, it doesn't matter),
                self.Table[j, 0] = 0
                for k in range(self.n_dim - 1):  # each vertex that the ants pass
                    # a point that has been passed and cannot be retaken
                    taboo_set = set(self.Table[j, :k + 1])
                    # a list of allowed vertices to select from,
                    allow_list = list(set(range(self.n_dim)) - taboo_set)
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum() #probability normalization,
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # distance calculation
            y = np.array([self.func(i) for i in self.Table])

            # fixing the best time
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            #counting the pheromone that will be added to the edge,
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # for each ant
                for k in range(self.n_dim - 1):  #for each peak
                    # ants move from vertex n1 to vertex n2,
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]
                    delta_tau[n1, n2] += 1 / y[j]  # application of pheromone
                # ants crawl from the last peak back to the first
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]
                delta_tau[n1, n2] += 1 / y[j]  # application of pheromene

            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    fit = run
    
# path length calculation,
def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def main():
    # creating ant colony with params (n_dim, size_pop=10, max_iter=20, distance_matrix=None, alpha=1, beta=2, rho=0.1)
    aca = ACO_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=s_pop,  # number of ants
                  max_iter=m_iter, 
                  distance_matrix=distance_matrix,
                  alpha=_alpha, #Alpha
                  beta=_beta,   #Beta
                  rho=_rho      #Rho
                  ) 

    best_x, best_y = aca.run()

    #Displaying results
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    for index in range(0, len(best_points_)):
        ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    ax[0].plot(best_points_coordinate[:, 0],
               best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    # resizing graphs
    plt.rcParams['figure.figsize'] = [25, 15]
    plt.show()

if __name__ == "__main__":
    start_time = time.time() # saving the start time of the execution
    main() # execution of code
    print("Timpul executare): %s seconds" %abs (time.time() - start_time)) # runtime calculation