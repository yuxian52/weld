import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 禁用GPU（如果需要）
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    
    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        X_scaler, y_scaler
    )

# 构建BP神经网络模型
def build_model():
    model = Sequential()
    
    # 输入层到第一个隐含层：3->6，使用ReLU激活函数
    model.add(Dense(6, input_dim=3, activation='relu'))
    
    # 第一个隐含层到第二个隐含层：6->4，使用ReLU激活函数
    model.add(Dense(4, activation='relu'))
    
    # 第二个隐含层到输出层：4->3，输出层采用线性函数
    model.add(Dense(3, activation='linear'))
    
    # 注意：TensorFlow的Adam优化器最接近LM算法的性能
    # 编译模型，使用均方误差作为损失函数
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mse']
    )
    
    return model

# 训练模型
def train_model(model, train_data, val_data, epochs=1000, batch_size=32, patience=50):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # 早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # 模型检查点回调
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    return model, history

# 评估模型
def evaluate_model(model, test_data, y_scaler):
    X_test, y_test = test_data
    
    # 在测试集上评估模型
    test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集均方误差 (MSE): {test_mse:.6f}")
    print(f"测试集均方根误差 (RMSE): {np.sqrt(test_mse):.6f}")
    
    # 进行预测
    y_pred_normalized = model.predict(X_test)
    
    # 将预测结果转换回原始尺度
    y_pred = y_scaler.inverse_transform(y_pred_normalized)
    y_true = y_scaler.inverse_transform(y_test)
    
    # 计算原始尺度下的误差
    mse_original = np.mean((y_pred - y_true) ** 2)
    print(f"原始尺度下的MSE: {mse_original:.6f}")
    
    # 计算每个输出维度的误差
    for i in range(3):
        dim_mse = np.mean((y_pred[:, i] - y_true[:, i]) ** 2)
        dim_rmse = np.sqrt(dim_mse)
        print(f"输出维度 {i+1} MSE: {dim_mse:.6f}, RMSE: {dim_rmse:.6f}")
    
    return y_pred, y_true

# 绘制训练过程中的损失曲线
def plot_loss_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练过程中的损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves_tensorflow.png')
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
    plt.savefig('predictions_vs_true_tensorflow.png')
    plt.show()

# 主函数
def main():
    # 文件路径
    file_path = 'WeldData.csv'
    
    # 加载和预处理数据
    print("正在加载和预处理数据...")
    train_data, val_data, test_data, X_scaler, y_scaler = load_and_preprocess_data(file_path)
    
    # 创建模型
    print("创建神经网络模型...")
    model = build_model()
    
    # 打印模型结构
    print("神经网络模型结构:")
    model.summary()
    
    # 训练模型
    print("开始训练模型...")
    model, history = train_model(model, train_data, val_data)
    
    # 绘制损失曲线
    plot_loss_curves(history)
    
    # 评估模型
    print("评估模型性能...")
    y_pred, y_true = evaluate_model(model, test_data, y_scaler)
    
    # 绘制预测结果对比图
    plot_predictions(y_pred, y_true)
    
    # 保存最终模型
    model.save('bp_model_tensorflow.h5')
    print("模型已保存为 'bp_model_tensorflow.h5'")
    print("最佳模型已保存为 'best_model.h5'")

if __name__ == "__main__":
    # 确保使用TensorFlow 2.x
    print(f"TensorFlow版本: {tf.__version__}")
    main()