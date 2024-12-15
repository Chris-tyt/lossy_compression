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
        
        # 根据数据规模自动设置种群大小
        if population_size is None:
            self.population_size = max(100, len(self.block_ids) * 3)
        else:
            self.population_size = population_size
        
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def create_individual(self) -> Dict[int, Tuple[int, float]]:
        """创建一个个体（一个可能的解）"""
        individual = {}
        for block_id in self.block_ids:
            individual[block_id] = random.choice(self.data[block_id])
        return individual
    
    def calculate_fitness(self, individual: Dict[int, Tuple[int, float]]) -> float:
        """计算适应度（越高越好）"""
        total_waves = sum(waves for waves, _ in individual.values())
        avg_mse = sum(mse for _, mse in individual.values()) / len(individual)
        
        # 如果不满足MSE约束，给予惩罚
        if avg_mse > 4e-5:
            return -float('inf')
        
        # 适应度为波数的负值（因为我们要最小化波数）
        return -total_waves
    
    def crossover(self, parent1: Dict[int, Tuple[int, float]], 
                 parent2: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        """交叉操作"""
        child = {}
        for block_id in self.block_ids:
            if random.random() < 0.5:
                child[block_id] = parent1[block_id]
            else:
                child[block_id] = parent2[block_id]
        return child
    
    def mutate(self, individual: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
        """变异操作"""
        for block_id in self.block_ids:
            if random.random() < self.mutation_rate:
                individual[block_id] = random.choice(self.data[block_id])
        return individual
    
    def optimize(self) -> Dict[int, Tuple[int, float]]:
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 选择最优的一部分个体
            elite = [ind for ind, _ in fitness_scores[:self.population_size//2]]
            
            # 生成新一代
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            # 打印当前最优解
            best_solution = fitness_scores[0][0]
            total_waves = sum(waves for waves, _ in best_solution.values())
            avg_mse = sum(mse for _, mse in best_solution.values()) / len(best_solution)
            print(f"Generation {generation + 1}: Total waves = {total_waves}, Avg MSE = {avg_mse}")
        
        return fitness_scores[0][0]

def find_optimal_combination():
    # 读取文件数据
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
    
    # 使用遗传算法优化
    optimizer = GeneticOptimizer(data)
    best_solution = optimizer.optimize()
    
    # 计算结果
    total_waves = sum(waves for waves, _ in best_solution.values())
    avg_mse = sum(mse for _, mse in best_solution.values()) / len(best_solution)
    
    # 创建结果列表
    result = []
    for block_id in sorted(best_solution.keys()):
        waves, _ = best_solution[block_id]
        result.append(waves)
    
    # 打印信息
    if avg_mse < 4e-5:
        print(f"\n找到可行解:")
        print(f"总波数: {total_waves}")
        print(f"平均MSE: {avg_mse}")
    else:
        print("未找到满足条件的解")
    
    return result

if __name__ == "__main__":
    result_list = find_optimal_combination()
    print("\n最终选择的波数列表:")
    print(result_list)
