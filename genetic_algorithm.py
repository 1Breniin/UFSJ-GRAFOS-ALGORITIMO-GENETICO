# genetic_algorithm.py

import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, graph, population_size=100, generations=500, mutation_rate=0.01):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = list(range(self.graph.num_nodes))
            random.shuffle(individual)
            population.append(individual)
        return population

    def fitness(self, individual):
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += self.graph.get_distance(individual[i], individual[i + 1])
        total_distance += self.graph.get_distance(individual[-1], individual[0])
        return 1 / total_distance  # Minimizar a distância

    def select(self):
        fitness_scores = [self.fitness(ind) for ind in self.population]
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        
        # Convert population to a numpy array
        population_array = np.array(self.population)
        
        # Select an individual based on the probabilities
        selected_index = np.random.choice(len(self.population), p=probabilities)
        return self.population[selected_index]
    
    def crossover(self, parent1, parent2):
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start + 1, len(parent1))
        
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        
        current_pos = end
        for gene in parent2:
            if gene not in child:
                if current_pos >= len(parent1):
                    current_pos = 0
                child[current_pos] = gene
                current_pos += 1
        
        return child
    
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
    
    def run(self):
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select()
                parent2 = self.select()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        
        best_individual = max(self.population, key=self.fitness)
        return best_individual, 1 / self.fitness(best_individual)
