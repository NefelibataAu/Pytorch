"""
深度学习训练计时工具
==================
专门用于您的RNN预测项目的训练时间统计
"""

import time
import datetime
import pandas as pd
import numpy as np

class TrainingTimer:
    """训练计时器"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.training_history = []
    
    def start_training(self, total_epochs):
        """开始训练计时"""
        self.start_time = time.perf_counter()
        self.epoch_times = []
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.training_history = []
        
        print(f"🚀 训练开始！总epochs: {total_epochs}")
        print(f"⏰ 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
    
    def epoch_start(self):
        """每个epoch开始时调用"""
        self.epoch_start_time = time.perf_counter()
    
    def epoch_end(self, loss, print_interval=10):
        """每个epoch结束时调用"""
        epoch_time = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.current_epoch += 1
        
        # 记录训练历史
        self.training_history.append({
            'epoch': self.current_epoch,
            'loss': loss,
            'epoch_time': epoch_time,
            'timestamp': datetime.datetime.now()
        })
        
        # 计算统计信息
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - self.current_epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        if self.current_epoch % print_interval == 0:
            print(f"Epoch [{self.current_epoch:3d}/{self.total_epochs}] "
                  f"Loss: {loss:.4f} "
                  f"Epoch时间: {epoch_time:.2f}s "
                  f"平均时间: {avg_epoch_time:.2f}s "
                  f"预计剩余: {estimated_remaining:.1f}s")
    
    def end_training(self):
        """训练结束"""
        total_time = time.perf_counter() - self.start_time
        
        print("\n" + "=" * 50)
        print("🎉 训练完成！")
        print(f"⏰ 结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总训练时间: {self._format_time(total_time)}")
        print(f"📊 总epochs: {self.total_epochs}")
        print(f"⚡ 平均每epoch时间: {np.mean(self.epoch_times):.2f}s")
        print(f"🐌 最慢epoch: {max(self.epoch_times):.2f}s")
        print(f"🚀 最快epoch: {min(self.epoch_times):.2f}s")
        print("=" * 50)
        
        return self.get_training_summary()
    
    def _format_time(self, seconds):
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
    
    def get_training_summary(self):
        """获取训练摘要"""
        if not self.training_history:
            return None
        
        df = pd.DataFrame(self.training_history)
        
        summary = {
            'total_time': time.perf_counter() - self.start_time,
            'total_epochs': len(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times),
            'min_epoch_time': min(self.epoch_times),
            'max_epoch_time': max(self.epoch_times),
            'final_loss': self.training_history[-1]['loss'],
            'best_loss': min([h['loss'] for h in self.training_history]),
            'training_df': df
        }
        
        return summary

# 使用示例和测试
def demo_training_timer():
    """演示训练计时器的使用"""
    print("训练计时器演示")
    print("=" * 40)
    
    # 创建计时器
    timer = TrainingTimer()
    
    # 开始训练
    total_epochs = 50
    timer.start_training(total_epochs)
    
    # 模拟训练过程
    for epoch in range(total_epochs):
        timer.epoch_start()
        
        # 模拟训练计算
        time.sleep(0.1)  # 模拟训练时间
        
        # 模拟损失值（逐渐下降）
        loss = 1.0 / (epoch + 1) + 0.01
        
        timer.epoch_end(loss, print_interval=10)
    
    # 结束训练
    summary = timer.end_training()
    
    # 显示训练历史
    if summary and 'training_df' in summary:
        print("\n训练历史（前5行）：")
        print(summary['training_df'].head())

# 实际应用到您的RNN代码
def integrate_with_rnn_training():
    """展示如何集成到您的RNN训练代码中"""
    print("\n\n集成到RNN训练的示例代码：")
    print("-" * 40)
    
    code_example = '''
# 在您的app_predict.py中集成计时器

from training_timer import TrainingTimer

# 创建计时器
timer = TrainingTimer()

# 在训练开始前
num_epochs = 100
timer.start_training(num_epochs)

# 修改您的训练循环
for epoch in range(num_epochs):
    timer.epoch_start()  # 开始计时
    
    # 原有的训练代码
    outputs = rnn(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    
    timer.epoch_end(loss.item())  # 结束计时并记录损失

# 训练结束后
summary = timer.end_training()
'''
    
    print(code_example)

if __name__ == "__main__":
    # 运行演示
    demo_training_timer()
    
    # 显示集成示例
    integrate_with_rnn_training()
