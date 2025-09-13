import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time 

torch.manual_seed(42)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层 -> 隐藏层
        self.relu = nn.ReLU()                          # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes) # 隐藏层 -> 输出层
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 初始化模型
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
print(model)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),                      # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))        # 归一化到[-1, 1]
])

# 下载数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root = './data', 
    train = True,
    download = True,
    transform = transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root = './data',
    train = False,
    transform = transform
)

# 创建数据加载器
batch_size = 100
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)

# 查看数据集信息
print("训练集大小:", len(train_dataset))
print("测试集大小:", len(test_dataset))
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for device in ['cuda', 'cpu']:

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()                           # 交叉熵损失（已包含Softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    total_step = len(train_loader)
    loss_history = []
    acc_history = []
    start = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式（启用Dropout/BatchNorm）
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # 将数据移动到设备
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算本epoch指标
        epoch_loss = running_loss / total_step
        epoch_acc = 100 * correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        # 打印训练进度
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Loss: {epoch_loss:.4f}, '
            f'Accuracy: {epoch_acc:.2f}%')

    end = time.perf_counter()
    print(f"{device} time : {end - start}")
    # 可视化训练过程
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(loss_history, label='Training Loss')
    plt.title('Loss Curve')
    plt.subplot(1,2,2)
    plt.plot(acc_history, label='Training Accuracy')
    plt.title('Accuracy Curve')
    plt.show()
    

model.eval()  # 设置为评估模式（关闭Dropout/BatchNorm）
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'测试集准确率: {100 * correct / total:.2f}%')

# 获取测试集样本
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.reshape(-1, 28*28).to(device)

# 预测结果
outputs = model(images)
_, preds = torch.max(outputs, 1)
preds = preds.cpu().numpy()
images = images.cpu().reshape(-1, 28, 28).numpy()

# 绘制预测结果
plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

#torch.save(model, 'fashion_mnist_model.pth')
#loaded_model = torch.load('fashion_mnist_model.pth')

torch.save(model.state_dict(), 'model_weights.pth')

# 加载时需要先创建相同结构的模型
new_model = SimpleNN(784, 128, 10).to(device)
new_model.load_state_dict(torch.load('model_weights.pth'))