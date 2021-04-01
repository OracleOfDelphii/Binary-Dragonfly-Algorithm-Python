%___________________________________________________________________%
%  Binary Dragonfly Algorithm (BDA) implented with Python3          %
%                                                                   %
%  This is a translation of BDA matlab codes available from:                                 %
%                https://seyedalimirjalili.com/da                   %
%                                                                   %
%                                                                   %                                                  %
%                                                                   %
%   S. Mirjalili, Dragonfly algorithm: a new meta-heuristic         %
%   optimization technique for solving single-objective, discrete,  %
%   and multi-objective problems, Neural Computing and Applications % 
%   DOI: http://dx.doi.org/10.1007/s00521-015-1920-1                %
%___________________________________________________________________%
from bda import BDA
from mycost import MyCost

#import matplotlib.pyplot as plt
#from numpy import linspace

CostFunction = MyCost

Max_iteration = 500 # Max it
N = 10 # number of particles
nVar = 20

A = BDA(N, Max_iteration, nVar, CostFunction)
print("Best pos:", A[0])
print("Best Score:", A[1])
print("Convergence curve:", A[2])
