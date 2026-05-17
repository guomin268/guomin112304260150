import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

val_split = 0.1
val_indices = int(len(X_train) * val_split)
X_val, y_val = X_train[:val_indices], y_train[:val_indices]
X_train_final, y_train_final = X_train[val_indices:], y_train[val_indices:]

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_model(optimizer, learning_rate, batch_size, data_augmentation=False, early_stopping=False):
    model = create_cnn_model()
    
    opt = None
    if optimizer == 'SGD':
        opt = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'AdamW':
        opt = optimizers.AdamW(learning_rate=learning_rate)
    
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks_list = []
    if early_stopping:
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        callbacks_list.append(early_stop)
    
    if data_augmentation:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(X_train_final)
        history = model.fit(datagen.flow(X_train_final, y_train_final, batch_size=batch_size),
                           epochs=30,
                           validation_data=(X_val, y_val),
                           callbacks=callbacks_list)
    else:
        history = model.fit(X_train_final, y_train_final,
                           batch_size=batch_size,
                           epochs=30,
                           validation_data=(X_val, y_val),
                           callbacks=callbacks_list)
    
    train_acc = model.evaluate(X_train_final, y_train_final, verbose=0)[1]
    val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    min_loss = min(history.history['loss'])
    conv_epoch = len(history.history['loss'])
    
    return history, train_acc, val_acc, test_acc, min_loss, conv_epoch, model

experiments = [
    {'name': 'Exp1', 'optimizer': 'SGD', 'lr': 0.01, 'batch_size': 64, 'aug': False, 'early_stop': False},
    {'name': 'Exp2', 'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 64, 'aug': False, 'early_stop': False},
    {'name': 'Exp3', 'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 128, 'aug': False, 'early_stop': True},
    {'name': 'Exp4', 'optimizer': 'Adam', 'lr': 0.001, 'batch_size': 64, 'aug': True, 'early_stop': True},
]

results = []
histories = {}
final_models = {}

print("开始执行对比实验...")
for exp in experiments:
    print(f"\n=== {exp['name']} ===")
    print(f"优化器: {exp['optimizer']}, 学习率: {exp['lr']}, BatchSize: {exp['batch_size']}")
    print(f"数据增强: {exp['aug']}, EarlyStopping: {exp['early_stop']}")
    
    history, train_acc, val_acc, test_acc, min_loss, conv_epoch, model = train_model(
        optimizer=exp['optimizer'],
        learning_rate=exp['lr'],
        batch_size=exp['batch_size'],
        data_augmentation=exp['aug'],
        early_stopping=exp['early_stop']
    )
    
    histories[exp['name']] = history
    final_models[exp['name']] = model
    
    results.append({
        '实验编号': exp['name'],
        'Train Acc': f"{train_acc:.4f}",
        'Val Acc': f"{val_acc:.4f}",
        'Test Acc': f"{test_acc:.4f}",
        '最低 Loss': f"{min_loss:.4f}",
        '收敛 Epoch': conv_epoch
    })
    
    print(f"结果: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}, Loss={min_loss:.4f}, Epoch={conv_epoch}")

results_df = pd.DataFrame(results)
print("\n=== 对比实验结果 ===")
print(results_df)

results_df.to_csv('../results/experiment_results.csv', index=False)
print("\n结果已保存到 results/experiment_results.csv")

plt.figure(figsize=(12, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, exp in enumerate(experiments):
    history = histories[exp['name']]
    plt.plot(history.history['loss'], label=f"{exp['name']} - 训练Loss", color=colors[i], linewidth=2)
    plt.plot(history.history['val_loss'], label=f"{exp['name']} - 验证Loss", color=colors[i], linewidth=2, linestyle='--')

plt.title('4组对比实验 Loss 曲线', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('../results/loss_comparison.png', dpi=300, bbox_inches='tight')
print("Loss对比曲线图已保存到 results/loss_comparison.png")

print("\n=== 训练最终提交模型 (AdamW + 数据增强) ===")
history, train_acc, val_acc, test_acc, min_loss, conv_epoch, final_model = train_model(
    optimizer='AdamW',
    learning_rate=0.001,
    batch_size=64,
    data_augmentation=True,
    early_stopping=True
)

final_model.save('../models/cnn_model.h5')
print(f"最终模型已保存到 models/cnn_model.h5")
print(f"最终模型准确率: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")

print("\n=== 生成 Kaggle 提交文件 ===")
test_df = pd.read_csv('c:/Users/26830/Downloads/333/test.csv')
X_kaggle = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
predictions = np.argmax(final_model.predict(X_kaggle), axis=1)

submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
submission.to_csv('../results/sample_submission.csv', index=False)
print("Kaggle提交文件已保存到 results/sample_submission.csv")

print("\n=== 实验完成 ===")
print("对比实验结果:")
print(results_df)
