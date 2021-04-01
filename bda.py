import numpy as np
from random import random
from math import sqrt
from numpy import inf

# row vector to Column vector
def rtc(X):
    if len(np.shape(X)) != 1 : print("Error")
    else:
        return X.reshape(-1,1)

def BDA(N, max_iter, nVar, CostFunction):
    
    print("the bda algorithm is optimizing your problem...")

    dim = nVar

    # where is X and DeltaX defined?
    X = np.zeros((nVar, N))
    DeltaX = np.zeros((nVar, N))
    Convergence_curve = np.zeros(max_iter)


    #  dim x 1 
    #Food_pos = np.zeros((dim, 1))
    #Enemy_pos = np.zeros((dim, 1))

    # (dim,) arrays
    Food_pos = np.zeros(dim)
    Enemy_pos = np.zeros(dim)

    
    for i in range(0, N):
        for j in range(1, nVar):
            if random() < 0.5:
                X[j][i] = 0
            else:
                X[j][i] = 1

            if random() <= 0.5:
                DeltaX[j][i] = 0
            else:
                DeltaX[j][i] = 1



    Fitness = np.zeros(N)
    Enemy_fitness = 0
    Food_fitness = 0
    for it in range(1, max_iter + 1):
        w = 0.9 - it * ((0.9 - 0.4) / max_iter)
        my_c = 0.1 - it * ((0.1 - 0) / (max_iter/2))

        if my_c < 0:
            my_c = 0

        s = 2 * random() * my_c # Seperation weight
        a = 2 * random() * my_c # Alignment Weight
        c = 2 * random() * my_c # Cohesion weight
        f = 2 * random() # Food attraction weight
        e = my_c  # Enemy distraction weight

        if it > (3 * max_iter / 4):
            e = 0

        for i in range(0, N):
            Fitness[i] = CostFunction(X[:, i])

            if Fitness[i] < Food_fitness:
                Food_fitness = Fitness[i]
                Food_pos = X[:, i]

            if Fitness[i] > Enemy_fitness:
                Enemy_fitness = Fitness[i]
                Enemy_pos = X[:, i]


        for i in range(0, N):
            index = 0
            neighbours_no = 0


            Neighbours_DeltaX = np.zeros((nVar, N))
            Neighbours_X = np.zeros((nVar, N))

            # Find the neighbouring solutions (all the dragonflies are
            # assumed as a group in binary search spaces
            for j in range(0, N):
                if (i != j):
                    index = index + 1
                    neighbours_no = neighbours_no + 1
                    Neighbours_DeltaX[:, index] = DeltaX[:, j]
                    Neighbours_X[:, index] = X[:, j]

            # Seperation Eq. (3.1)
            S = np.zeros(dim)
            for k in range(0, neighbours_no):
                S = S + (Neighbours_X[:, k] - X[:, i])

            S = -S

            # Alignment Eq. (3.2)

            A = np.transpose(np.sum(np.transpose(Neighbours_DeltaX), axis=0)) / neighbours_no

            # Coheision Eq. (3.3)

            C_temp = np.transpose(np.sum(np.transpose(Neighbours_X), axis=0)) / neighbours_no

            C = C_temp - X[:, i]

            # Attraction to food Eq. (3.4)
            F = Food_pos - X[:, i]

            # Distraction from Enemy
            E = Enemy_pos + X[:, i]

            for j in range(0, dim):
                DeltaX[j][i] = (s * S[j] + 
                        a * A[j] + 
                        c * C[j] + 
                        f * F[j] + 
                        e * E[j] + 
                        w * DeltaX[j][i])
                if DeltaX[j][i] > 6:
                    DeltaX[j][i] = 6


                # Eq. (3.11)
                T = abs(DeltaX[j][i] / sqrt((
                    1 + DeltaX[j][i] ** 2))) # V3 transfer function
                # Eq. (3.12)
                if random() < T: # Equation (10)
                    if X[j][i] == 0:
                        X[j][i] = 1
                    else:
                        X[j][i] = 0

            Convergence_curve[it] = Food_fitness
            Best_pos = Food_pos
            Best_score = Food_fitness

            return (Best_pos, Best_score, Convergence_curve)



