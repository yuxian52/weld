import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据加载和预处理
def load_and_preprocess_data(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 分离输入和输出数据
    X = data.iloc[:, :3].values  # 前3列作为输入
    y = data.iloc[:, 3:].values  # 后3列作为输出
    
    # 数据归一化处理（输入数据标准化至[0,1]区间）
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))  # 输出数据也进行归一化便于训练
    
    X_normalized = X_scaler.fit_transform(X)
    y_normalized = y_scaler.fit_transform(y)
    
    # 划分数据集：训练集（70%）、验证集（15%）与测试集（15%）
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=15/85, random_state=42  # 15/(70+15) = 15/85
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    return (
        (X_train_tensor, y_train_tensor),
        (X_val_tensor, y_val_tensor),
        (X_test_tensor, y_test_tensor),
        X_scaler, y_scaler
    )

# 定义BP神经网络模型
class BPNeuralNetwork(nn.Module):
    def __init__(self):
        super(BPNeuralNetwork, self).__init__()
        # 输入层到第一个隐含层：3->6
        self.fc1 = nn.Linear(3, 6)
        # 第一个隐含层到第二个隐含层：6->4
        self.fc2 = nn.Linear(6, 4)
        # 第二个隐含层到输出层：4->3
        self.fc3 = nn.Linear(4, 3)
        # ReLU激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 隐含层采用ReLU激活函数
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # 输出层采用线性函数（不使用激活函数）
        x = self.fc3(x)
        return x

# 训练模型
def train_model(model, train_data, val_data, learning_rate=0.01, epochs=1000, patience=50):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # 定义损失函数（均方误差）
    criterion = nn.MSELoss()
    
    # 使用Adam优化器（由于PyTorch没有直接提供LM算法，使用Adam作为替代）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练过程的损失值
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # 记录损失值
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停在第 {epoch+1} 轮")
                break
        
        # 打印训练进度
        if (epoch+1) % 100 == 0:
            print(f'轮次 [{epoch+1}/{epochs}], 训练损失: {loss.item():.6f}, 验证损失: {val_loss.item():.6f}')
    
    # 加载最佳模型参数
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses

# 评估模型
def evaluate_model(model, test_data, y_scaler):
    X_test, y_test = test_data
    
    model.eval()
    with torch.no_grad():
        y_pred_normalized = model(X_test)
    
    # 将预测结果转换回原始尺度
    y_pred = y_scaler.inverse_transform(y_pred_normalized.numpy())
    y_true = y_scaler.inverse_transform(y_test.numpy())
    
    # 计算测试集上的均方误差
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"测试集均方误差 (MSE): {mse:.6f}")
    print(f"测试集均方根误差 (RMSE): {rmse:.6f}")
    
    # 计算每个输出维度的误差
    for i in range(3):
        dim_mse = np.mean((y_pred[:, i] - y_true[:, i]) ** 2)
        dim_rmse = np.sqrt(dim_mse)
        print(f"输出维度 {i+1} MSE: {dim_mse:.6f}, RMSE: {dim_rmse:.6f}")
    
    return y_pred, y_true

# 绘制训练过程中的损失曲线
def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程中的损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()

# 绘制预测结果与真实值的对比图
def plot_predictions(y_pred, y_true):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i in range(3):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        axes[i].plot([y_true[:, i].min(), y_true[:, i].max()], 
                    [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        axes[i].set_title(f'输出维度 {i+1} 预测 vs 真实')
        axes[i].set_xlabel('真实值')
        axes[i].set_ylabel('预测值')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_true.png')
    plt.show()

# 主函数
def main():
    # 文件路径
    file_path = 'WeldData.csv'
    
    # 加载和预处理数据
    print("正在加载和预处理数据...")
    train_data, val_data, test_data, X_scaler, y_scaler = load_and_preprocess_data(file_path)
    
    # 创建模型
    model = BPNeuralNetwork()
    print("神经网络模型结构:")
    print(model)
    
    # 训练模型
    print("开始训练模型...")
    model, train_losses, val_losses = train_model(model, train_data, val_data, learning_rate=0.01)
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses)
    
    # 评估模型
    print("评估模型性能...")
    y_pred, y_true = evaluate_model(model, test_data, y_scaler)
    
    # 绘制预测结果对比图
    plot_predictions(y_pred, y_true)
    
    # 保存模型
    torch.save(model.state_dict(), 'bp_model.pth')
    print("模型已保存为 'bp_model.pth'")

if __name__ == "__main__":
    main()