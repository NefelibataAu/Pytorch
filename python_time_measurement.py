"""
Python时间统计方法详解
====================
演示各种时间测量方法，特别适用于机器学习和深度学习
"""

import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

print("Python时间统计方法详解")
print("=" * 60)

# =============================================================================
# 1. 基本时间测量方法
# =============================================================================
print("\n1. 基本时间测量方法")
print("-" * 40)

# 1.1 time.time() - 时间戳
print("1.1 time.time() - 获取当前时间戳")
start_time = time.time()
time.sleep(0.1)  # 模拟操作
end_time = time.time()
elapsed = end_time - start_time
print(f"操作耗时: {elapsed:.4f} 秒")

# 1.2 time.perf_counter() - 高精度计时器
print("\n1.2 time.perf_counter() - 高精度计时器（推荐）")
start = time.perf_counter()
time.sleep(0.1)
end = time.perf_counter()
elapsed = end - start
print(f"高精度耗时: {elapsed:.6f} 秒")

# 1.3 time.process_time() - CPU时间
print("\n1.3 time.process_time() - CPU处理时间")
start = time.process_time()
sum([i**2 for i in range(10000)])  # CPU密集型操作
end = time.process_time()
elapsed = end - start
print(f"CPU处理时间: {elapsed:.6f} 秒")

# =============================================================================
# 2. 装饰器方式计时
# =============================================================================
print("\n\n2. 装饰器方式计时")
print("-" * 40)

def timing_decorator(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.4f} 秒")
        return result
    return wrapper

@timing_decorator
def heavy_computation(n):
    """模拟重计算"""
    return sum(i**2 for i in range(n))

result = heavy_computation(100000)
print(f"计算结果: {result}")

# =============================================================================
# 3. 上下文管理器方式
# =============================================================================
print("\n\n3. 上下文管理器方式")
print("-" * 40)

class Timer:
    """计时上下文管理器"""
    def __init__(self, name="操作"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        print(f"{self.name} 耗时: {elapsed:.4f} 秒")

# 使用上下文管理器
with Timer("数据处理"):
    data = [i**2 for i in range(50000)]
    result = sum(data)

# =============================================================================
# 4. 机器学习训练时间统计
# =============================================================================
print("\n\n4. 机器学习训练时间统计")
print("-" * 40)

# 4.1 训练循环时间统计
def train_with_timing():
    """带时间统计的训练函数"""
    # 模拟数据
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # 简单模型
    model = nn.Linear(10, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100
    total_start = time.perf_counter()
    
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        
        # 前向传播
        forward_start = time.perf_counter()
        outputs = model(X)
        forward_time = time.perf_counter() - forward_start
        
        # 计算损失
        loss_start = time.perf_counter()
        loss = criterion(outputs, y)
        loss_time = time.perf_counter() - loss_start
        
        # 反向传播
        backward_start = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.perf_counter() - backward_start
        
        epoch_time = time.perf_counter() - epoch_start
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                  f"Epoch时间={epoch_time:.4f}s, "
                  f"前向={forward_time:.4f}s, "
                  f"反向={backward_time:.4f}s")
    
    total_time = time.perf_counter() - total_start
    print(f"\n总训练时间: {total_time:.4f} 秒")
    print(f"平均每轮时间: {total_time/epochs:.4f} 秒")
    return total_time

# 执行训练
train_time = train_with_timing()

# =============================================================================
# 5. 批量操作时间统计
# =============================================================================
print("\n\n5. 批量操作时间统计")
print("-" * 40)

def batch_processing_timing():
    """批量处理时间统计"""
    data_sizes = [1000, 5000, 10000, 50000]
    
    for size in data_sizes:
        with Timer(f"处理 {size} 条数据"):
            # 模拟数据处理
            data = np.random.randn(size, 100)
            result = np.mean(data, axis=1)
            processed = result[result > 0]

batch_processing_timing()

# =============================================================================
# 6. 性能分析工具
# =============================================================================
print("\n\n6. 性能分析工具")
print("-" * 40)

# 6.1 使用cProfile进行性能分析
import cProfile
import pstats
from io import StringIO

def performance_test():
    """性能测试函数"""
    # 模拟复杂计算
    result = []
    for i in range(1000):
        temp = []
        for j in range(100):
            temp.append(i * j + np.sin(i) + np.cos(j))
        result.append(sum(temp))
    return sum(result)

print("6.1 cProfile性能分析:")
profiler = cProfile.Profile()
profiler.enable()
result = performance_test()
profiler.disable()

# 输出性能统计
s = StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.sort_stats('cumulative')
ps.print_stats(10)  # 显示前10个最耗时的函数
print(s.getvalue())

# =============================================================================
# 7. 实际应用：训练时间预估
# =============================================================================
print("\n\n7. 实际应用：训练时间预估")
print("-" * 40)

class TrainingTimer:
    """训练时间预估器"""
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.current_epoch = 0
    
    def start_training(self):
        """开始训练"""
        self.start_time = time.perf_counter()
        self.epoch_times = []
        self.current_epoch = 0
        print("训练开始...")
    
    def epoch_end(self, total_epochs):
        """每个epoch结束时调用"""
        if self.current_epoch == 0:
            self.epoch_times.append(time.perf_counter() - self.start_time)
        else:
            self.epoch_times.append(time.perf_counter() - self.start_time - sum(self.epoch_times[:-1]))
        
        self.current_epoch += 1
        
        if self.current_epoch >= 2:  # 至少2个epoch才能预估
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = total_epochs - self.current_epoch
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            print(f"Epoch {self.current_epoch}/{total_epochs}: "
                  f"当前epoch时间={self.epoch_times[-1]:.2f}s, "
                  f"平均epoch时间={avg_epoch_time:.2f}s, "
                  f"预计剩余时间={estimated_remaining:.2f}s")

# 使用训练时间预估器
timer = TrainingTimer()
timer.start_training()

for epoch in range(5):
    # 模拟训练过程
    time.sleep(0.2)  # 模拟训练时间
    timer.epoch_end(5)

# =============================================================================
# 8. 时间格式化和显示
# =============================================================================
print("\n\n8. 时间格式化和显示")
print("-" * 40)

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} 分 {secs:.1f} 秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} 小时 {minutes} 分 {secs:.1f} 秒"

# 测试时间格式化
test_times = [45.5, 125.8, 3661.2]
for t in test_times:
    print(f"{t} 秒 = {format_time(t)}")

# =============================================================================
# 9. 实际项目中的应用建议
# =============================================================================
print("\n\n9. 实际项目中的应用建议")
print("-" * 40)

print("""
时间统计最佳实践：

1. 选择合适的时间函数：
   - time.perf_counter(): 高精度，适合性能测试
   - time.process_time(): CPU时间，适合CPU密集型任务
   - time.time(): 简单场景

2. 训练时间统计：
   - 记录总训练时间
   - 记录每个epoch时间
   - 预估剩余时间
   - 分析瓶颈（前向/反向传播）

3. 代码组织：
   - 使用装饰器简化计时
   - 使用上下文管理器
   - 创建专门的计时类

4. 性能优化：
   - 使用cProfile找出瓶颈
   - 比较不同实现的性能
   - 监控内存使用情况

5. 日志记录：
   - 将时间信息写入日志
   - 使用pandas记录训练历史
   - 可视化训练时间趋势
""")

print("\n" + "=" * 60)
print("Python时间统计方法演示完成！")
print("=" * 60)
