import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import io

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

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.flatten().reshape(1, 784)
    return img_array, img_array.reshape(28, 28)

def main():
    st.set_page_config(page_title="手写数字识别", page_icon="🔢", layout="wide")
    
    st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .tab-content {
        padding: 1rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction {
        font-size: 3rem;
        font-weight: bold;
    }
    .confidence {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="title">🔢 手写数字识别系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">上传图片或在画板上书写数字进行识别</div>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        with st.spinner('正在加载模型...'):
            st.session_state.model = train_model()
    
    tab1, tab2 = st.tabs(["📷 上传图片", "✏️ 手写画板"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("选择手写数字图片", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='原始图片', use_column_width=True)
            
            processed_img, preview_img = preprocess_image(image)
            
            with col2:
                st.image(preview_img, caption='预处理后 (28x28)', use_column_width=True, clamp=True)
            
            if st.session_state.model is not None:
                prediction = st.session_state.model.forward(processed_img)
                predicted_num = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                top3_indices = np.argsort(-prediction[0])[:3]
                top3_probs = prediction[0][top3_indices] * 100
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction">识别结果: {predicted_num}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">置信度: {confidence:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("Top-3 预测结果")
                for i, idx in enumerate(top3_indices):
                    st.write(f"{i+1}. 数字 {int(idx)}: {top3_probs[i]:.2f}%")
            
            else:
                st.error('模型加载失败')
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        if 'canvas' not in st.session_state:
            st.session_state.canvas = Image.new('L', (280, 280), color=255)
            st.session_state.drawing = False
            st.session_state.last_pos = None
        
        canvas_placeholder = st.empty()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            stroke_width = st.slider("笔画粗细", 5, 20, 10)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("识别"):
                    small_canvas = st.session_state.canvas.resize((28, 28))
                    processed, _ = preprocess_image(small_canvas)
                    if st.session_state.model is not None:
                        prediction = st.session_state.model.forward(processed)
                        predicted_num = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        
                        st.session_state.prediction = predicted_num
                        st.session_state.confidence = confidence
                        st.session_state.top3 = np.argsort(-prediction[0])[:3]
                        st.session_state.top3_probs = prediction[0][st.session_state.top3] * 100
            
            with col_b:
                if st.button("清空"):
                    st.session_state.canvas = Image.new('L', (280, 280), color=255)
                    if 'prediction' in st.session_state:
                        del st.session_state.prediction
        
        with col2:
            canvas_bytes = io.BytesIO()
            st.session_state.canvas.save(canvas_bytes, format='PNG')
            canvas_bytes.seek(0)
            
            uploaded = st.file_uploader("绘制或上传", type=["png"], key="canvas_upload")
            
            if uploaded is not None:
                st.session_state.canvas = Image.open(uploaded).convert('L')
            
            canvas_image = st.image(canvas_bytes, use_column_width=True)
            
            if 'prediction' in st.session_state:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction">识别结果: {st.session_state.prediction}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">置信度: {st.session_state.confidence:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("Top-3 预测")
                for i, idx in enumerate(st.session_state.top3):
                    st.write(f"{i+1}. 数字 {int(idx)}: {st.session_state.top3_probs[i]:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
