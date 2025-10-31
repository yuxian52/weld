import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from typing import List, Tuple, Dict

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义BP神经网络模型（与训练时使用的结构相同）
class BPNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(BPNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(3, 6)
        self.fc2 = torch.nn.Linear(6, 4)
        self.fc3 = torch.nn.Linear(4, 3)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载数据和归一化器
def load_data_and_scalers(file_path: str) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """加载数据并创建用于归一化的scaler对象"""
    data = pd.read_csv(file_path)
    X = data.iloc[:, :3].values  # 前3列作为输入
    y = data.iloc[:, 3:].values  # 后3列作为输出
    
    # 创建归一化器并拟合数据范围
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaler.fit(X)
    y_scaler.fit(y)
    
    return X, y, X_scaler, y_scaler

# 遗传算法类
class GeneticAlgorithm:
    def __init__(self, model: BPNeuralNetwork, X_scaler: MinMaxScaler, y_scaler: MinMaxScaler,
                 X_bounds: np.ndarray, population_size: int = 100, generations: int = 50,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.model = model
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.X_bounds = X_bounds  # 输入参数的上下界
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.chromosome_length = 3  # 输入参数数量
        
        # 存储每一代的最优解和平均适应度
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution_history = []
    
    def initialize_population(self) -> np.ndarray:
        """初始化种群，随机生成在参数范围内的个体"""
        population = np.zeros((self.population_size, self.chromosome_length))
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                population[i, j] = random.uniform(self.X_bounds[j, 0], self.X_bounds[j, 1])
        return population
    
    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """评估种群中每个个体的适应度
        目标：第5列输出最大化、第6列输出最小化
        """
        fitness_scores = np.zeros(self.population_size)
        
        # 对种群中的每个个体进行评估
        for i in range(self.population_size):
            # 将输入参数归一化
            input_params = population[i].reshape(1, -1)
            input_normalized = self.X_scaler.transform(input_params)
            input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
            
            # 使用模型预测输出
            self.model.eval()
            with torch.no_grad():
                output_normalized = self.model(input_tensor)
            
            # 将输出转换回原始尺度
            output_original = self.y_scaler.inverse_transform(output_normalized.numpy())
            
            # 提取第5列和第6列输出（索引为1和2，因为输出有3列）
            output_col5 = output_original[0, 1]  # 第5列（output2）
            output_col6 = output_original[0, 2]  # 第6列（output3）
            
            # 计算适应度分数：第5列最大化，第6列最小化
            # 这里使用加权和的方式，权重可以根据实际需求调整
            # 为了平衡两个目标，我们可以对它们进行标准化处理
            # 例如：适应度 = 0.6 * 归一化的第5列值 - 0.4 * 归一化的第6列值
            
            # 简单的适应度计算（可以根据实际需求调整权重）
            # 假设我们希望第5列和第6列具有相同的重要性
            fitness_scores[i] = output_col5 - output_col6
        
        return fitness_scores
    
    def selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """选择操作，使用轮盘赌选择法"""
        # 适应度可能为负，所以需要调整
        fitness_offset = np.maximum(0, -np.min(fitness_scores)) + 1e-6
        adjusted_fitness = fitness_scores + fitness_offset
        
        # 计算选择概率
        total_fitness = np.sum(adjusted_fitness)
        selection_probs = adjusted_fitness / total_fitness
        
        # 进行选择
        selected_indices = np.random.choice(range(self.population_size), 
                                           size=self.population_size, 
                                           p=selection_probs)
        
        return population[selected_indices]
    
    def crossover(self, population: np.ndarray) -> np.ndarray:
        """交叉操作，使用单点交叉"""
        new_population = population.copy()
        
        for i in range(0, self.population_size, 2):
            if random.random() < self.crossover_rate and i + 1 < self.population_size:
                # 选择交叉点
                crossover_point = random.randint(1, self.chromosome_length - 1)
                
                # 交换基因片段
                new_population[i, crossover_point:], new_population[i+1, crossover_point:] = \
                    new_population[i+1, crossover_point:].copy(), new_population[i, crossover_point:].copy()
        
        return new_population
    
    def mutation(self, population: np.ndarray) -> np.ndarray:
        """变异操作，对每个基因进行小概率变异"""
        new_population = population.copy()
        
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                if random.random() < self.mutation_rate:
                    # 在参数范围内进行小幅随机变异
                    mutation_range = (self.X_bounds[j, 1] - self.X_bounds[j, 0]) * 0.1
                    mutation = random.uniform(-mutation_range, mutation_range)
                    new_population[i, j] += mutation
                    
                    # 确保变异后的值仍在范围内
                    new_population[i, j] = max(self.X_bounds[j, 0], 
                                             min(self.X_bounds[j, 1], new_population[i, j]))
        
        return new_population
    
    def run(self) -> Dict:
        """运行遗传算法进行优化"""
        # 初始化种群
        population = self.initialize_population()
        
        # 开始进化过程
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = self.evaluate_fitness(population)
            
            # 记录当前一代的最优解
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx].copy()
            
            # 记录历史数据
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_scores))
            self.best_solution_history.append(best_solution.copy())
            
            # 输出当前进度
            print(f"第 {generation+1} 代 - 最佳适应度: {best_fitness:.6f} - 平均适应度: {np.mean(fitness_scores):.6f}")
            
            # 执行遗传操作
            population = self.selection(population, fitness_scores)
            population = self.crossover(population)
            population = self.mutation(population)
        
        # 获取最终的最优解
        final_best_idx = np.argmax(self.best_fitness_history)
        final_best_solution = self.best_solution_history[final_best_idx]
        final_best_fitness = self.best_fitness_history[final_best_idx]
        
        # 使用最优解进行预测
        best_input_tensor = torch.tensor(self.X_scaler.transform(final_best_solution.reshape(1, -1)), 
                                       dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            best_output_normalized = self.model(best_input_tensor)
        best_output = self.y_scaler.inverse_transform(best_output_normalized.numpy())[0]
        
        return {
            'best_solution': final_best_solution,
            'best_fitness': final_best_fitness,
            'best_output': best_output,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }
    
    def plot_results(self):
        """绘制遗传算法优化结果"""
        # 绘制适应度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='最佳适应度')
        plt.plot(self.avg_fitness_history, label='平均适应度')
        plt.title('遗传算法优化过程中的适应度变化')
        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.legend()
        plt.grid(True)
        plt.savefig('ga_fitness_curve.png')
        plt.show()

def main():
    # 文件路径
    file_path = 'd:\\PyCode\\JiaoCai\\JiaoCai\\上传\\WeldData.csv'
    model_path = 'd:\\PyCode\\JiaoCai\\JiaoCai\\bp_model.pth'  # 训练好的模型路径
    
    # 加载数据和创建归一化器
    X, y, X_scaler, y_scaler = load_data_and_scalers(file_path)
    
    # 确定输入参数的上下界（基于原始数据范围）
    X_bounds = np.zeros((3, 2))
    for i in range(3):
        X_bounds[i, 0] = np.min(X[:, i])  # 下界
        X_bounds[i, 1] = np.max(X[:, i])  # 上界
    
    # 加载训练好的模型
    model = BPNeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("已加载训练好的BP神经网络模型")
    
    # 设置遗传算法参数
    ga = GeneticAlgorithm(
        model=model,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        X_bounds=X_bounds,
        population_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # 运行遗传算法优化
    print("开始使用遗传算法优化输入参数...")
    results = ga.run()
    
    # 输出优化结果
    print("\n优化完成！")
    print(f"最优输入参数组合:")
    print(f"  input1: {results['best_solution'][0]:.4f}")
    print(f"  input2: {results['best_solution'][1]:.4f}")
    print(f"  input3: {results['best_solution'][2]:.4f}")
    print(f"\n对应的输出预测:")
    print(f"  output1: {results['best_output'][0]:.4f}")
    print(f"  output2 (第5列): {results['best_output'][1]:.4f}")
    print(f"  output3 (第6列): {results['best_output'][2]:.4f}")
    print(f"\n最佳适应度值: {results['best_fitness']:.6f}")
    
    # 绘制优化结果
    ga.plot_results()
    
    # 绘制参数重要性分析
    plt.figure(figsize=(12, 5))
    
    # 绘制最优解的参数分布
    plt.subplot(1, 2, 1)
    param_names = ['input1', 'input2', 'input3']
    plt.bar(param_names, results['best_solution'])
    plt.title('最优输入参数值')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 绘制输出预测结果
    plt.subplot(1, 2, 2)
    output_names = ['output1', 'output2', 'output3']
    plt.bar(output_names, results['best_output'])
    plt.title('对应的输出预测值')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ga_optimization_results.png')
    plt.show()

if __name__ == "__main__":
    main()