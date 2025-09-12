import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(2, 3)  # 子模块 → 注册到 _modules
        self.fc1 = nn.Linear(3, 4)
        self.weight = nn.Parameter(torch.randn(3))  # 参数 → 注册到 _parameters
        self.register_buffer("running_mean", torch.zeros(3))  # buffer → 注册到 _buffers
        self.register_buffer("non_persistent_buffer", torch.zeros(3), persistent=False)

model = MyModel()

print(model._parameters)
print(model._modules)
print(model._buffers)
print(model._non_persistent_buffers_set)
print(model._state_dict_hooks)

# print(model.fc.weight)
# print(model.weight)
# print("参数:", list(model.named_parameters()), '\n')
# print("buffer:", list(model.named_buffers()))
# print("子模块:", list(model.named_children()))
# print("state_dict:", model.state_dict().keys())




