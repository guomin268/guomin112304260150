import gradio as gr
import numpy as np
import pandas as pd
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

def train_model():
    train_path = 'c:/Users/26830/Downloads/333/train.csv'
    if not os.path.exists(train_path):
        return None
    
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop('label', axis=1).values / 255.0
    y_train = train_df['label'].values
    
    model = SimpleNN(784, 128, 10)
    
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
    
    epochs = 10
    batch_size = 64
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            grads = compute_gradients(model, X_batch, y_batch)
            for p, g in zip(params, grads):
                p.grad = g
            optimizer.step()
    
    return model

model = train_model()

def preprocess_image(image):
    if image is None:
        return None
    
    image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.flatten().reshape(1, 784)
    return img_array

def predict_digit(image):
    if model is None:
        return "模型加载失败", [], []
    
    processed = preprocess_image(image)
    if processed is None:
        return "请上传图片", [], []
    
    prediction = model.forward(processed)
    predicted_num = np.argmax(prediction)
    confidence = prediction[0] * 100
    
    top3_indices = np.argsort(-confidence)[:3]
    top3_labels = [int(i) for i in top3_indices]
    top3_probs = [float(confidence[i]) for i in top3_indices]
    
    return f"识别结果: {predicted_num}", top3_labels, top3_probs

def create_probability_bar(probabilities):
    digits = list(range(10))
    bars = []
    for i, prob in enumerate(probabilities):
        bars.append(gr.Bar(label=str(i), value=prob))
    return bars

with gr.Blocks(title="手写数字识别", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔢 手写数字识别系统")
    gr.Markdown("上传手写数字图片或在画板上直接书写，我来帮你识别！")
    
    with gr.Tabs():
        with gr.TabItem("📷 上传图片"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="上传手写数字图片")
                    upload_btn = gr.Button("开始识别", variant="primary")
                
                with gr.Column():
                    result_output = gr.Label(label="识别结果", show_label=True)
                    top3_output = gr.Textbox(label="Top-3 预测")
    
        with gr.TabItem("✏️ 手写画板"):
            with gr.Row():
                with gr.Column():
                    sketch_input = gr.Sketchpad(label="在此处书写数字", shape=(280, 280))
                    sketch_btn = gr.Button("识别手写数字", variant="primary")
                    clear_btn = gr.Button("清空画板")
                
                with gr.Column():
                    sketch_result = gr.Label(label="识别结果", show_label=True)
                    sketch_top3 = gr.Textbox(label="Top-3 预测")
                    prob_plot = gr.BarChart(label="概率分布")
    
    def handle_upload(image):
        result, top3_labels, top3_probs = predict_digit(image)
        top3_text = "\n".join([f"{label}: {prob:.2f}%" for label, prob in zip(top3_labels, top3_probs)])
        return result, top3_text
    
    def handle_sketch(image):
        result, top3_labels, top3_probs = predict_digit(image)
        top3_text = "\n".join([f"{label}: {prob:.2f}%" for label, prob in zip(top3_labels, top3_probs)])
        
        processed = preprocess_image(image)
        if processed is not None:
            probs = model.forward(processed)[0] * 100
            prob_data = {"digit": list(range(10)), "probability": probs.tolist()}
        else:
            prob_data = {"digit": [], "probability": []}
        
        return result, top3_text, prob_data
    
    def clear_sketch():
        return None, "", "", {"digit": [], "probability": []}
    
    upload_btn.click(handle_upload, inputs=image_input, outputs=[result_output, top3_output])
    sketch_btn.click(handle_sketch, inputs=sketch_input, outputs=[sketch_result, sketch_top3, prob_plot])
    clear_btn.click(clear_sketch, outputs=[sketch_input, sketch_result, sketch_top3, prob_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
