from mimetypes import init
import numpy as np
from enum import Enum

DEBUG = 1

class PopulationGenerationMethod(Enum):
    Uniform = 1
    Random = 2

def get_population(gen_method: PopulationGenerationMethod, size: int, d: int, dimension: int):
    return [  ]

def genetic(A: np.matrix, B: np.matrix, c: float, initial_population_size: int, d: int):
    dimension = len(A)
    population = get_population(PopulationGenerationMethod.Random, initial_population_size, d, dimension) 


def main():
    A = None
    B = None
    c = None
    d = None
    initial_population_size = None

    if DEBUG == 0:
        dimension = int(input("Enter the dimension of \'A\' matrix:"))
        if dimension < 1:
            print("Bad dimension! Exiting...")
            exit(1)

        A = np.ones(dimension * dimension).reshape(dimension, dimension)
        for y in range(dimension):
            for x in range(dimension):
                A[y][x] = float(input("Enter A[" + str(y) + "][" + str(x) + "]..."))

        B = np.ones(dimension).reshape(dimension, 1)
        for i in range(dimension):
            B[i][0] = float(input("Enter B[" + str(i) + "]..."))
        
        c = float(input("Enter c..."))

        initial_population_size = int(input("Enter initial population size..."))
        if initial_population_size < 0:
            print("Bad initial population size! Exiting...")
            exit(1)

        d = int(input("Enter d..."))
        if d < 0:
            print("Bad d! Exiting...")
            exit(1)
    else:
        dimension = 2
        A = np.array([1, 1, 0, 1]).reshape(dimension, dimension)
        B = np.array([5, -2]).reshape(dimension, 1)
        c = 0
        d = 3
        initial_population_size = 100
    solution = genetic(A, B, c, initial_population_size, d)


if __name__ == '__main__':
    main()
