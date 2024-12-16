import random
import numpy as np
from typing import Dict, List, Tuple

# file_path = 'text.txt'
file_path = 'optimization_results.txt'

class GeneticOptimizer:
    def __init__(self, data: Dict[int, List[Tuple[int, float]]], 
                 population_size: int = None,
                 generations: int = 5000,
                 mutation_rate: float = 0.1):
        self.data = data
        self.block_ids = sorted(data.keys())
        
        # Automatically set population size based on data scale
        if population_size is None:
            self.population_size = max(100, len(self.block_ids) * 3)
        else:
            self.population_size = population_size
        
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def create_individual(self) -> Dict[int, Tuple[int, float]]:
        """Create an individual (a possible solution)"""
        individual = {}
        for block_id in self.block_ids:
            individual[block_id] = random.choice(self.data[block_id])
        return individual
    
    def calculate_fitness(self, individual: Dict[int, Tuple[int, float]]) -> float:
        """Calculate fitness (the higher, the better)"""
        total_waves = sum(waves for waves, _ in individual.values())
        avg_mse = sum(mse for _, mse in individual.values()) / len(individual)
        
        # If MSE constraint is not met, apply a penalty
        if avg_mse > 4e-5:
            return -float('inf')
        
        # Fitness is the negative value of total waves (since we want to minimize waves)
        return -total_waves
    
    def crossover(self, parent1: Dict[int, Tuple[int, float]], 
                 parent2: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        """Crossover operation"""
        child = {}
        for block_id in self.block_ids:
            if random.random() < 0.5:
                child[block_id] = parent1[block_id]
            else:
                child[block_id] = parent2[block_id]
        return child
    
    def mutate(self, individual: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        """Mutation operation"""
        for block_id in self.block_ids:
            if random.random() < self.mutation_rate:
                individual[block_id] = random.choice(self.data[block_id])
        return individual
    
    def optimize(self) -> Dict[int, Tuple[int, float]]:
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Calculate fitness
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select the best individuals
            elite = [ind for ind, _ in fitness_scores[:self.population_size//2]]
            
            # Generate the next generation
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            # Print current best solution
            best_solution = fitness_scores[0][0]
            total_waves = sum(waves for waves, _ in best_solution.values())
            avg_mse = sum(mse for _, mse in best_solution.values()) / len(best_solution)
            print(f"Generation {generation + 1}: Total waves = {total_waves}, Avg MSE = {avg_mse}")
        
        return fitness_scores[0][0]

def find_optimal_combination():
    # Read file data
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            block_id, waves, mse = line.strip().split(',')
            block_id = int(block_id)
            waves = int(waves)
            mse = float(mse)
            
            if block_id not in data:
                data[block_id] = []
            data[block_id].append((waves, mse))
    
    # Use genetic algorithm to optimize
    optimizer = GeneticOptimizer(data)
    best_solution = optimizer.optimize()
    
    # Calculate results
    total_waves = sum(waves for waves, _ in best_solution.values())
    avg_mse = sum(mse for _, mse in best_solution.values()) / len(best_solution)
    
    # Create result list
    result = []
    for block_id in sorted(best_solution.keys()):
        waves, _ = best_solution[block_id]
        result.append(waves)
    
    # Print information
    if avg_mse < 4e-5:
        print(f"\nFound feasible solution:")
        print(f"Total waves: {total_waves}")
        print(f"Average MSE: {avg_mse}")
    else:
        print("No feasible solution found")
    
    return result

if __name__ == "__main__":
    result_list = find_optimal_combination()
    print("\nFinal selected wave number list:")
    print(result_list)
