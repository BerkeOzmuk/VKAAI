import csv
import random
import matplotlib.pyplot as plt
import time

def import_csv(filename):
    """Imports a CSV file and returns its contents as a matrix."""
    matrix = []
    with open(filename, newline='') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(read_csv), next(read_csv)
        for row in read_csv:
            matrix.append([row[0]] + [float(x) for x in row[1:]])

    return matrix

matrix = import_csv('RondeTafel.csv') #Matrix with the knights and affinities 

def fitness_function(individual):
    """Calculates the fitness of an individual based on the affinity matrix.
    -The fitness is determined by the sum of affinities between consecutive knights in the individual list.
    -The affinities are looked up in a global matrix where each knight has an affinity with every other knight.
    -At last the sum of the affinity is returned"""

    sum_affinity = 0
    for indexA in range(len(individual)):
        indexB = (indexA + 1) % len(individual) 
        knightA = individual[indexA]
        knightB = individual[indexB]

        sum_affinity += matrix[knightA][knightB + 1] * matrix[knightB][knightA + 1]
        
    return sum_affinity

def orderbased_crossover(male_parent, female_parent):
    """Performs order-based crossover between two parents a male and female to produce a child.
    -First it chooses two random genes (positions). These random genes will be used to perform the crossover
    -Then after the crossover it will return the child with the new order of genes"""
    male_parent_length = len(male_parent)

    random_genes = sorted(random.sample(range(male_parent_length), 2)) #https://www.geeksforgeeks.org/python-random-sample-function/

    child = [-1] * male_parent_length
    
    for positions in random_genes:
        child[positions] = male_parent[positions]
    
    current_position = 0
    for gene in female_parent:
        if gene not in child:
            while child[current_position] != -1:
                current_position += 1
            child[current_position] = gene
    
    return child

def insertion_mutation(individual):
    """Performs an insertion mutation on an individual.
    -The mutation randomly selects a gene from the individual.
    -Removes the gene from its original position.
    -Inserts it back into a new randomly chosen position within the individual."""
    gene = random.choice(individual)
    individual.pop(individual.index(gene))
    individual.insert(random.choice(range(len(individual))), gene) #https://pynative.com/python-random-choice/

    return individual

def evolve(population, population_size, retain=0.2, random_select=0.05, mutate=0.01):
    """Evolves a population of individuals (genetic algorithm).
    -This function performs the selection, crossover and mutation to create new generations of individuals from the current population."""
    ranksIndividuals = [ (fitness_function(individual), individual) for individual in population ]
    ranksIndividuals = [ individual[1] for individual in sorted(ranksIndividuals) ]

    num_elitist_parents = int(len(ranksIndividuals) * retain)
    num_random_parents = int(len(ranksIndividuals) * random_select)

    population = ranksIndividuals[num_elitist_parents:]

    for _ in range(0, num_random_parents):
        population.append(random.choice(ranksIndividuals[num_elitist_parents:]))

    while(len(population) < population_size):
        male_parent = random.choice(ranksIndividuals[:num_elitist_parents])
        female_parent = random.choice(ranksIndividuals[num_elitist_parents:])
        population.append(orderbased_crossover(male_parent, female_parent))

    for individual in population:
        if(mutate > random.random()):
            insertion_mutation(individual)

    return population

def print_knights(best_table):
    """Prints the knights in a certain layout"""
    for indexA in range(len(best_table)):
        indexB = (indexA + 1) % len(best_table) 
        knightA = best_table[indexA]
        knightB = best_table[indexB]

        print(matrix[knightA][0], "(", matrix[knightA][knightB + 1], "X", matrix[knightB][knightA + 1], ")", matrix[knightB][0])
            
def main():
    """This is the main :p"""
    start_time = time.time()
    knights = [0,1,2,3,4,5,6,7,8,9,10,11]
    random.seed(0)
    population_size = 100
    population = []

    ranksIndividuals = []
    affinities = []
    best_fitness = []
    best_individuals = []

    for _ in range(0, population_size):
        individual = knights.copy()
        random.shuffle(individual)
        population.append(individual)
    
    for _ in range(64):
        population = evolve(population, population_size)
        individuals = [ (fitness_function(individual), individual) for individual in population ]
        ranksIndividuals = [ individual[1] for individual in sorted(individuals) ]
        affinities = [ individual[0] for individual in sorted(individuals) ]
        best_individuals.append([affinities[-1], ranksIndividuals[-1]])
    
    end_time = time.time()
    elapsed_time = end_time - start_time 

    for epoch, individual in enumerate(best_individuals):
        print("Epoch: " + str(epoch) + " Fitness: " + str(individual[0]) + " Individual: " + str(individual[1])) #individual[0] is fitness and individual[1] the list table order
    
    print("Elapsed time: " + str(elapsed_time))

    print_knights(best_individuals[-1][1])    
    
    for i in best_individuals:
        best_fitness.append(i[0])

    plt.plot(range(0, len(best_fitness)), best_fitness)
    plt.xlabel('Epoch')
    plt.ylabel('Fitness')
    plt.show()

if __name__ == "__main__":
    main()