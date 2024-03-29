"""
function GENETIC-ALGORITHM(population, FITNESS-FN) returns an individual
    inputs: population, a set of individuals
            FITNESS-FN, a function that measures the fitness of an individual

repeat
    new population ←empty set
    for i = 1 to SIZE(population) do
        x ←RANDOM-SELECTION(population, FITNESS-FN)
        y ←RANDOM-SELECTION(population, FITNESS-FN)
        child ←REPRODUCE(x , y)
        if (small random probability) then child ←MUTATE(child )
        add child to new population
    population ←new population
until some individual is fit enough, or enough time has elapsed
return the best individual in population, according to FITNESS-FN

function REPRODUCE(x , y) returns an individual
    inputs: x , y, parent individuals
n←LENGTH(x ); c←random number from 1 to n
return APPEND(SUBSTRING(x, 1, c), SUBSTRING(y, c + 1, n))

"""
from random import uniform
from random import random, seed
from random import randint
import matplotlib.pyplot as plt

'''
Board:
list of 8 ints where the the index+1 is the x value or column
and where the int value is the y-value or the row.
'''
x_axis = list()
y_axis = list()
selection = dict()
fitness = dict()


def generateRandomMember():
    board = list()
    for i in range(8):
        value = randint(1, 8)
        board.append(value)
    # Board is Built
    return tuple(board)


def generatePopulation(size):
    pop = list()
    for i in range(size):
        b = generateRandomMember()
        while b in pop:
            b = generateRandomMember()
        pop.append(generateRandomMember())
    return pop


maxFitness = 28


def diagonal_clashes(board):
    clashes = 0
    for i in range(0, 8):
        x = i + 1
        y = board[i]
        counter = 0
        for j in range(x, 8):
            counter += 1
            xi = j + 1
            yi = board[j]
            if yi is y + counter or yi is y - counter:
                clashes += 1
                break
    return clashes


'''
Fitness Function:
    f(board-state) = 24 - (# of clashes)
'''


def calcFitness(b):
    clashes = 0
    # there should not be any doubled up in columns

    for i in range(8):
        y = b[i]
        # check rows
        for j in range(i + 1, 8):
            if y is b[j]:
                clashes += 1
                break
    # check diagonal
    clashes += diagonal_clashes(b)
    f = maxFitness - clashes
    return f


def buildFitnessFunc(population):
    for b in population:
        if b in fitness.keys():
            continue
        f = calcFitness(b)
        fitness[b] = f


def buildSelectionMap(pop, fitness_function):
    summation = 0
    size = len(pop)
    for state in pop:
        summation += fitness_function[state]
    for i in range(0, size):
        fi = fitness_function[pop[i]]
        si = fi / summation
        selection[pop[i]] = si


def sortSelect(b):
    return selection[b]


def sortPopInSelectionOrder(pop):
    new_pop = list(pop)
    new_pop.sort(key=sortSelect)
    return new_pop


def get_prob_intervals(pop):  # pop needs to come in sorted for this to work
    probs = list()
    # build probability interval
    size = len(pop)
    previous_probability = 0
    for i in range(0, size):
        b = pop[i]
        p = previous_probability + selection[b]
        interval = (previous_probability, p)
        probs.append(interval)
        previous_probability = p
    return probs


def parentSelection(population):  # roulette wheel selection method
    # generate random value in [0, 1]
    pop = sortPopInSelectionOrder(population)
    # pop = population
    x = random()
    probs = get_prob_intervals(pop)
    size = len(pop)
    assert size == len(probs)
    for j in range(0, size):
        interval = probs[j]
        lower = interval[0]
        upper = interval[1]
        if x < upper:  # then choose this state as a parent
            return pop[j]
    assert False  # should not reach here


def Reproduce(x, y):
    n = len(x)
    c = randint(0, n - 1)
    p1 = list(x)
    p2 = list(y)
    child = p1[0:c] + p2[c:n]
    return tuple(child)


def ChanceMutate():
    x = random()
    if x > .9:
        return True
    return False


def Mutate(board):
    new_child = list(board)
    gene = randint(0, 7)
    new_value = randint(1, 8)
    new_child[gene] = new_value
    return tuple(new_child)


def select_best(pop):
    maxIndex = 0
    for j in range(1, len(pop)):
        if fitness[pop[maxIndex]] <= fitness[pop[j]]:
            maxIndex = j
    return pop[maxIndex]


def average_fitness(pop):
    summation = 0
    for state in pop:
        summation += fitness[state]
    return summation / len(pop)


def GeneticAlgorithm(p, maxIter):
    counter = 1
    pop = list(p)
    while counter <= maxIter:
        x_axis.append(counter)
        new_population = list()
        size = len(pop)
        while len(new_population) != size:
            x = parentSelection(pop)
            y = parentSelection(pop)
            child = Reproduce(x, y)
            if ChanceMutate():
                child = Mutate(child)
            # check if already in new_pop
            # if child not in new_population:
            new_population.append(child)
            # check if solution
            if calcFitness(child) is maxFitness:
                # this is the last population that needs generated
                counter += maxIter
        pop = new_population
        buildFitnessFunc(pop)
        y = average_fitness(pop)
        y_axis.append(y)
        buildSelectionMap(pop, fitness)
        # pop = sortPopInSelectionOrder(pop)
        counter += 1
    return select_best(pop), pop


if __name__ == '__main__':
    seed()
    Iterations = 500
    # generate population
    PopulationCount = 1000
    population = generatePopulation(PopulationCount)
    buildFitnessFunc(population)
    buildSelectionMap(population, fitness)
    ###########

    answer, final_population = GeneticAlgorithm(population, Iterations)
    if answer is not None:
        if calcFitness(answer) == maxFitness:
            print("The answer generated: " + str(answer))
        else:
            print("No solution was found. Here is the best Solution Generated:")
            print("Solution: " + str(answer))
            print("Fitness Level: " + str(calcFitness(answer)) + " out of " + str(maxFitness))

        print("Here is a random sample board from the initial population:")
        r = randint(0, len(population))
        print("Sequence: " + str(population[r]))
        score = calcFitness(population[r])
        print("Score for this State: " + str(score))
        print("\nHere is a random sample board from the final population:")
        print("Sequence: " + str(final_population[r]))
        score = calcFitness(final_population[r])
        print("Score for this State: " + str(score))
        # plotting the points
        plt.plot(x_axis, y_axis)
        # naming the x axis
        plt.xlabel('Population Generation Number')
        # naming the y axis
        plt.ylabel('Average Fitness of Population')
        # giving a title to my graph
        plt.title('Population Size: ' + str(PopulationCount))
        # function to show the plot
        plt.show()
    else:
        print("No answer was found.")

    # test that they were generated and fitness func seems reasonable
