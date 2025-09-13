import torch

# 数据处理参数
DATA_ROOT = './data'
BATCH_SIZE = 64

# 训练模型参数
LR = 0.001
NUM_EPOCHS = 10

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)