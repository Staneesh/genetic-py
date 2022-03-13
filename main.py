from hashlib import new
from mimetypes import init
from multiprocessing import pool
import os
import numpy as np
from enum import Enum
import random as rd

DEBUG = 1

class PopulationGenerationMethod(Enum):
    Uniform = 1
    Random = 2

def get_population(gen_method: PopulationGenerationMethod, size: int, d: int, dimension: int):
    left = -pow(2, d)
    right = -left
    lerpi = lambda l, r, t : int( (1 - t) * l + t * r + 0.5 )

    population = [np.matrix([rd.randint(left, right) if gen_method == PopulationGenerationMethod.Random \
                   else lerpi(left, right, i/size) for j in range(dimension)]) for i in range(size) ]
    return population

def convert_to_binary(samples: list, bits_needed: int):
    res = []
    for x in samples:
        res.append([ np.binary_repr(xi, width=bits_needed) for xi in x.tolist()[0] ] )
    
    resi = res
    for i in range(len(res)):
        for j in range(len(res[i])):
            resi[i][j] = [ int(c) for c in res[i][j] ]
    return resi

def bool2int(x, bits): #1101
    num = 0
    for i in range(len(x)):
        if(x[i]):
            num += pow(2, bits - i - 1)
    if(x[0]):
        num -= 2 * pow(2, bits - 1)
    return num

def convert_from_binary(samples: list, bits_needed: int):
    resi = []
    for i in samples:
        res = [(bool2int(x, bits_needed)) for x in i]
        resi.append(np.matrix(res))
    return resi

def target_value(A: np.matrix, B: np.matrix, c: float, x: np.matrix):
    return (x.T * A * x + B.T * x + c).item()

def evaluate_population(A: np.matrix, B: np.matrix, c: float, x: list):
    return [ target_value(A, B, c, xi.transpose()) for xi in x ]

def upper_genes(bits: list):
    return bits[:int(len(bits) / 2)]

def lower_genes(bits: list):
    return bits[int(len(bits) / 2):]

def crossover(two_kids: list):
    crossed = []
    for i in range(len(two_kids[0])):
        crossed1 = lower_genes(two_kids[ 1 ][i]) + upper_genes(two_kids[ 0 ][i])
        crossed2 = lower_genes(two_kids[ 0 ][i]) + upper_genes(two_kids[ 1 ][i])
        crossed.append([crossed1, crossed2])
    return crossed

def get_new_population(parents: list, crossover_p: float, mutation_p: float):
    kids = []
    for m in range(len(parents)):
        matka = parents[m]
        for o in range(m, len(parents)):
            ojczym = parents[o]
            if matka != ojczym:
                dzieciuchy = [matka, ojczym]
                do_we_crossover = np.random.uniform(0, 1) >= 1 - crossover_p
                if do_we_crossover:
                    dzieciuchy = crossover(dzieciuchy)
                #do_we_mutate = np.random.uniform(0, 1) >= 1 - mutation_p
                for d in dzieciuchy:
                    kids.append(d)
    return kids

def genetic(A: np.matrix, B: np.matrix, c: float, initial_population_size: int, d: int):
    dimension = len(A)
    population = get_population(PopulationGenerationMethod.Random, initial_population_size, d, dimension) 
    global_max = -1000000000000
    for loop_index in range(100):
        evaluated = evaluate_population(A, B, c, population)
        for v in evaluated:
            global_max = max(global_max, v)

        x_y = [ [population[ i ], evaluated[ i ] ] for i in range(len(population)) ]
        x_y.sort(key = lambda row: -row[1])
        K_BEST = 4
        best_samples = [ x[ 0 ] for x in x_y[:K_BEST] ]
        CROSSOVER_P = 0.7
        MUTATION_P = 0.1
        best_samples = convert_to_binary(best_samples, d + 2)
        new_population = get_new_population(best_samples, CROSSOVER_P, MUTATION_P) #should not escape 2^-d, 2^d bounds!
        population = convert_from_binary(new_population, d + 2)
        #print("AFter conv:", population)
    return global_max

def main():
    os.system('cls')
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
        #A = np.array([1, 1, 0, 1]).reshape(dimension, dimension)
        #B = np.array([5, -2]).reshape(dimension, 1)
        A = np.matrix([[-1, 1], [0, 1]])
        B = np.matrix([5, -2]).transpose()
        c = 0
        d = 2
        initial_population_size = 100
    solution = genetic(A, B, c, initial_population_size, d)
    print(solution)

if __name__ == '__main__':
    main()
