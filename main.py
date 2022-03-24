# TODO:
# - FIFO
# - mutations
# - roulette

from cgitb import small
from hashlib import new
from mimetypes import init
from multiprocessing import pool
import matplotlib.pyplot as plt
import os
from sys import prefix
import numpy as np
from enum import Enum
import random as rd
from matplotlib import cm

DEBUG = 1


class PopulationGenerationMethod(Enum):
    Uniform = 1
    Random = 2


def get_population(gen_method: PopulationGenerationMethod, size: int, d: int, dimension: int):
    left = -pow(2, d)
    right = -left
    def lerpi(l, r, t): return int((1 - t) * l + t * r + 0.5)

    population = [np.matrix([rd.randint(left, right) if gen_method == PopulationGenerationMethod.Random
                             else lerpi(left, right, i/size) for j in range(dimension)]) for i in range(size)]
    return population


def convert_to_binary(samples: list, bits_needed: int):
    res = []
    for x in samples:
        res.append([np.binary_repr(xi, width=bits_needed)
                   for xi in x.tolist()[0]])

    resi = res
    for i in range(len(res)):
        for j in range(len(res[i])):
            resi[i][j] = [int(c) for c in res[i][j]]
    return resi


def bool2int(x, bits):  # 1101
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


def target_value(A: np.matrix, B: np.matrix, c: int, x: np.matrix):
    return (x.T * A * x + B.T * x + c).item()


def evaluate_population(A: np.matrix, B: np.matrix, c: int, x: list):
    return [target_value(A, B, c, xi.transpose()) for xi in x]


def upper_genes(bits: list):
    return bits[:int(len(bits) / 2)]


def lower_genes(bits: list):
    return bits[int(len(bits) / 2):]

def mutate(kid):
    vectorIndex = np.random.choice(range(0, len(kid)))
    bitIndex = np.random.choice(range(0, len(kid[0])))
    kid[vectorIndex][bitIndex] = 1 - kid[vectorIndex][bitIndex]
    return kid

def crossover(two_kids: list):
    crossed = [[], []]
    for i in range(len(two_kids[0])):
        crossed1 = lower_genes(two_kids[1][i]) + upper_genes(two_kids[0][i])
        crossed2 = lower_genes(two_kids[0][i]) + upper_genes(two_kids[1][i])
        crossed[0].append(crossed1)
        crossed[1].append(crossed2)
    #print("X", crossed)
    return crossed


def get_new_population(parents: list, crossover_p: float, mutation_p: float, initial_population_size: int):
    kids = []
    for m in range(len(parents)):
        matka = parents[m]
        for o in range(m, len(parents)):
            ojczym = parents[o]
            #if matka != ojczym:
            if True:
                dzieciuchy = [matka, ojczym]
                do_we_crossover = np.random.uniform(0, 1) >= 1 - crossover_p
                if do_we_crossover:
                    dzieciuchy = crossover(dzieciuchy)
                #do_we_mutate = np.random.uniform(0, 1) >= 1 - mutation_p
                for d in dzieciuchy:
                    isMutating = np.random.uniform(0, 1) >= 1 - mutation_p
                    if isMutating:
                        d = mutate(d)
                    kids.append(d)
                    if len(kids) == initial_population_size:
                        return kids
    
    return kids


def genetic(A: np.matrix, B: np.matrix, c: int, initial_population_size: int, d: int, loops: int, CROSSOVER_P: float, MUTATION_P: float):
    dimension = len(A)
    population = get_population(
        PopulationGenerationMethod.Random, initial_population_size, d, dimension)
    global_max = -1000000000000

    ax = None
    if dimension == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
       

    for loop_index in range(loops):
        if loop_index % int(np.sqrt(loops)) == 0:
            print("loop", loop_index,"/",loops,"...")
        evaluated = evaluate_population(A, B, c, population)
        for v in evaluated:
            global_max = max(global_max, v)

        x_y = [[population[i], evaluated[i]] for i in range(len(population))]
        x_y.sort(key=lambda row: -row[1])
    
        if dimension == 2:
            x = [x[0].tolist()[0][0] for x in x_y]
            y = [x[0].tolist()[0][1] for x in x_y]
            z = [x[1] for x in x_y]
            ax.scatter(x, y, z, label=loop_index, s=150, cmap=cm.coolwarm)
            ax.legend()
        
        #print("XY SIZE", x_y)
        parents = []
        for _ in range(initial_population_size):
            sorted_values = [int(knot[1]) for knot in x_y]
            #print("OG SORTED", sorted_values)
            shift_factor = 0
            if sorted_values[len(sorted_values) - 1] < 0:
                shift_factor = 2 * abs(sorted_values[len(sorted_values) - 1])
            sorted_values = [v + shift_factor for v in sorted_values]
            #print("SHIFTED SORTED", sorted_values)
            suffix_sums = [sum(sorted_values[i:]) for i in range(len(sorted_values))]
            #print("SUFFIX ", suffix_sums)

            choice = None
            if (suffix_sums[0] == 0):
                choice = 0
            else:
                choice = np.random.choice(range(1, suffix_sums[0] + 1))
            #print("RANODM CHOICE", choice)
            bigger_than_choice = [v for v in suffix_sums if v >= choice]
            choice_index = len(bigger_than_choice) - 1
            roulette_selection = x_y[choice_index]
            #print("SELECTION:", roulette_selection)
            parents.append(roulette_selection[0])

        parents = convert_to_binary(parents, d + 2)
        # should not escape 2^-d, 2^d bounds!
        new_population = get_new_population(
            parents, CROSSOVER_P, MUTATION_P, initial_population_size)
        #print(new_population)
        population = convert_from_binary(new_population, d + 2)
        #print("AFter conv:", population)
    if dimension == 2:
        solution_x = x_y[0][0].tolist()[0][0]
        solution_y = x_y[0][0].tolist()[0][1]
        X = np.arange(solution_x-20, solution_x+20, 0.5)
        Y = np.arange(solution_y-20, solution_y+20, 0.5)
        X, Y = np.meshgrid(X, Y)
        Z = c
        Z += B.tolist()[0][0]*X
        Z += B.tolist()[1][0]*Y
        #print(A.tolist())
        Z += A.tolist()[0][0]*X**2
        Z += X*Y*(A.tolist()[0][1]+A.tolist()[1][0])
        Z += A.tolist()[1][1]*Y**2
        
        ax.plot_wireframe(X, Y, Z)
        plt.show()
    return global_max, x_y


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
                A[y][x] = int(
                    input("Enter A[" + str(y) + "][" + str(x) + "]..."))

        B = np.ones(dimension).reshape(dimension, 1)
        for i in range(dimension):
            B[i][0] = int(input("Enter B[" + str(i) + "]..."))

        c = int(input("Enter c..."))

        initial_population_size = int(
            input("Enter initial population size..."))
        if initial_population_size < 0:
            print("Bad initial population size! Exiting...")
            exit(1)

        d = int(input("Enter d..."))
        if d < 0:
            print("Bad d! Exiting...")
            exit(1)

        loops = int(input("Enter loops limit..."))
        if loops < 0:
            print("Bad loops! Exiting...")
            exit(1)

        CROSSOVER_P = float(input("Enter crossover probability ..."))
        if CROSSOVER_P < 0 or CROSSOVER_P > 1:
            print("Bad crossover probability! Exiting...")
            exit(1)

        MUTATION_P = float(input("Enter mutation probability ..."))
        if MUTATION_P < 0 or MUTATION_P > 1:
            print("Bad mutation probability! Exiting...")
            exit(1)
    else:
        dimension = 2
        #A = np.array([1, 1, 0, 1]).reshape(dimension, dimension)
        #B = np.array([5, -2]).reshape(dimension, 1)
        
        #TEST: 2D example, expected outcome == 13
        #A = np.matrix([[-1, -1], [0, -1]])
        #B = np.matrix([5, -2]).transpose()
        #TEST: 3D example, expected outcome == 0
        A = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        B = np.matrix([0, 0, 0]).transpose()
        c = 0
        d = 5
        initial_population_size = 30
        loops = 10
        CROSSOVER_P = 0.7
        MUTATION_P = 0.1
    solution, last_population = genetic(A, B, c, initial_population_size, d, loops, CROSSOVER_P, MUTATION_P)
    print("SOLUTION =", solution)
    print("Last population (vector of pairs - each pair is (x, y), where x is a 'standing' vector and y is a number)")
    print("We use WIFO (as permitted), so the points are sorted by y's.")
    print("We will plot it if the dimension of A is 2. It may lag if loops is too large.")
    for i, knot in enumerate(last_population):
        print(i + 1, ": ", knot)


if __name__ == '__main__':
    main()
