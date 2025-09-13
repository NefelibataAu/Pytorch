from model import SimpleCNN
from data import get_dataloaders
from config import LR, NUM_EPOCHS, DEVICE
import torch
import matplotlib.pyplot as plt

train_loader, test_loader = get_dataloaders()

model = SimpleCNN().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr = LR)

total_step = len(train_loader)
loss_history = []
acc_history = []

for epoch in range(NUM_EPOCHS):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (data, label) in enumerate(train_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        
        output = model(data)
        loss = criterion(output, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        _, predict = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predict == label).sum().item()
    
    
    epoch_loss = running_loss / total_step
    epoch_acc = 100 * correct / total
    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)
    print(f"Echo[{epoch+1}/{NUM_EPOCHS}] : {epoch_acc}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_history, label='Training Loss')
plt.title('Loss Curve')
plt.subplot(1,2,2)
plt.plot(acc_history, label='Training Accuracy')
plt.title('Accuracy Curve')
plt.show()

model.eval()  # 关闭 dropout / batchnorm
with torch.no_grad():  # 不计算梯度
    for data, label in test_loader:
        data, label = data.to(DEVICE), label.to(DEVICE)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
accuracy = 100 * correct / total



