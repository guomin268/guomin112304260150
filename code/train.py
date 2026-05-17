import pandas as pd
import numpy as np
import os

class AdamWOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)

def train_and_predict():
    print("加载数据...")
    
    train_path = 'c:/Users/26830/Downloads/333/train.csv'
    test_path = 'c:/Users/26830/Downloads/333/test.csv'
    
    if not os.path.exists(train_path):
        print(f"错误: 训练数据文件不存在: {train_path}")
        return
    
    if not os.path.exists(test_path):
        print(f"错误: 测试数据文件不存在: {test_path}")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('label', axis=1).values / 255.0
    y_train = train_df['label'].values
    X_test = test_df.values / 255.0
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    print("\n创建简单神经网络模型...")
    
    class SimpleNN:
        def __init__(self, input_size, hidden_size, output_size):
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros(output_size)
        
        def relu(self, x):
            return np.maximum(0, x)
        
        def softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = self.relu(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            return self.softmax(self.z2)
    
    model = SimpleNN(784, 128, 10)
    
    print("\n优化器: AdamW")
    print("AdamW代码位置: 第11-31行")
    
    def one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]
    
    def compute_gradients(model, X, y):
        batch_size = X.shape[0]
        y_onehot = one_hot(y)
        
        output = model.forward(X)
        
        dz2 = output - y_onehot
        dW2 = (model.a1.T @ dz2) / batch_size
        db2 = np.mean(dz2, axis=0)
        
        da1 = dz2 @ model.W2.T
        dz1 = da1 * (model.z1 > 0)
        dW1 = (X.T @ dz1) / batch_size
        db1 = np.mean(dz1, axis=0)
        
        return [dW1, db1, dW2, db2]
    
    class Param:
        def __init__(self, data):
            self.data = data
            self.grad = None
    
    params = [Param(model.W1), Param(model.b1), Param(model.W2), Param(model.b2)]
    optimizer = AdamWOptimizer(params, lr=0.001, weight_decay=0.01)
    
    print("\n训练模型...")
    epochs = 10
    batch_size = 64
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            grads = compute_gradients(model, X_batch, y_batch)
            
            for p, g in zip(params, grads):
                p.grad = g
            
            optimizer.step()
            
            output = model.forward(X_batch)
            loss = -np.mean(np.log(output[np.arange(len(y_batch)), y_batch] + 1e-8))
            total_loss += loss * len(X_batch)
        
        avg_loss = total_loss / len(X_train)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print("\n预测测试集...")
    predictions = np.argmax(model.forward(X_test), axis=1)
    
    print("保存预测结果...")
    os.makedirs('../results', exist_ok=True)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    output_path = '../results/sample_submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")
    
    return submission

if __name__ == '__main__':
    train_and_predict()
