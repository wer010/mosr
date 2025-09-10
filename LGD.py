import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt

print("=" * 80)
print("L2L (Learning to Learn) vs MAML (Model-Agnostic Meta-Learning)")
print("=" * 80)


# ==================== L2L 实现 ====================
class L2LOptimizer(nn.Module):
    """
    L2L: 学习优化器本身
    目标：替代传统优化器 (SGD, Adam等)
    """

    def __init__(self, input_dim=1, hidden_dim=64):
        super(L2LOptimizer, self).__init__()

        print("\n🔵 L2L 优化器初始化:")
        print("   - 目标: 学习如何优化 (学习优化器)")
        print("   - 输入: [梯度, 参数, 历史信息]")
        print("   - 输出: 参数更新量")

        # 处理输入特征 [gradient, parameter, momentum等]
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # grad, param, momentum
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # LSTM维护优化历史
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # 输出更新量
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # 优化器状态
        self.hidden_state = None
        self.momentum = None

    def forward(self, gradients, parameters):
        """
        L2L的核心：学习如何根据梯度更新参数
        """
        batch_size = gradients.size(0)

        # 初始化动量
        if self.momentum is None:
            self.momentum = torch.zeros_like(parameters)

        # 更新动量
        self.momentum = 0.9 * self.momentum + gradients

        # 构建输入特征
        input_features = torch.cat([gradients, parameters, self.momentum], dim=-1)

        # 处理输入
        processed = self.input_processor(input_features)

        # 初始化LSTM状态
        if self.hidden_state is None:
            self.hidden_state = (
                torch.zeros(1, batch_size, 64),
                torch.zeros(1, batch_size, 64)
            )

        # LSTM前向传播
        # Disable CuDNN for higher-order gradients (meta-learning compatibility)
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, self.hidden_state = self.lstm(processed.unsqueeze(1), self.hidden_state)

        # 输出更新量
        update = self.output_layer(lstm_out.squeeze(1))

        return update

    def reset_state(self):
        """重置优化器状态"""
        self.hidden_state = None
        self.momentum = None


# ==================== MAML 实现 ====================
class SimpleModel(nn.Module):
    """
    MAML中的基础模型
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MAML:
    """
    MAML: 学习模型的初始化
    目标：找到好的初始参数，使得能快速适应新任务
    """

    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

        print("\n🔴 MAML 初始化:")
        print("   - 目标: 学习模型初始化 (学习模型参数)")
        print("   - 内循环: 快速适应单个任务")
        print("   - 外循环: 优化初始化参数")
        print(f"   - 内循环学习率: {inner_lr}")
        print(f"   - 元学习率: {meta_lr}")

    def inner_loop(self, task_data, task_labels, num_steps=5):
        """
        MAML内循环：在单个任务上快速适应
        """
        # 复制当前参数
        fast_weights = {}
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()

        # 内循环更新
        for step in range(num_steps):
            # 前向传播
            logits = self.functional_forward(task_data, fast_weights)
            loss = F.mse_loss(logits, task_labels)

            # 计算梯度
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True, retain_graph=True)

            # 更新fast_weights
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = param - self.inner_lr * grad

        return fast_weights

    def functional_forward(self, x, weights):
        """使用给定权重的函数式前向传播"""
        # 简化实现，实际中需要更复杂的权重管理
        x = F.linear(x, weights['net.0.weight'], weights['net.0.bias'])
        x = F.relu(x)
        x = F.linear(x, weights['net.2.weight'], weights['net.2.bias'])
        x = F.relu(x)
        x = F.linear(x, weights['net.4.weight'], weights['net.4.bias'])
        return x

    def meta_update(self, task_batch):
        """
        MAML外循环：元更新
        """
        meta_loss = 0

        for task_data, task_labels in task_batch:
            # 内循环适应
            fast_weights = self.inner_loop(task_data, task_labels)

            # 在adapted weights上计算损失
            adapted_logits = self.functional_forward(task_data, fast_weights)
            task_loss = F.mse_loss(adapted_logits, task_labels)
            meta_loss += task_loss

        # 元梯度更新
        meta_loss = meta_loss / len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


# ==================== 对比实验 ====================
def compare_l2l_vs_maml():
    """
    详细对比L2L和MAML的差异
    """
    print("\n" + "=" * 60)
    print("详细对比实验")
    print("=" * 60)

    # 创建简单的优化问题
    def create_quadratic_task(a, b, c):
        """创建二次函数任务 f(x) = ax² + bx + c"""

        def objective(x):
            return a * x ** 2 + b * x + c

        def gradient(x):
            return 2 * a * x + b

        return objective, gradient

    # ==================== L2L 实验 ====================
    print("\n🔵 L2L 实验:")
    print("场景: 学习如何优化不同的二次函数")

    l2l_optimizer = L2LOptimizer(input_dim=1, hidden_dim=32)
    l2l_meta_optimizer = optim.Adam(l2l_optimizer.parameters(), lr=0.001)

    # L2L元训练
    l2l_losses = []
    for epoch in range(100):
        epoch_loss = 0

        # 采样多个任务
        for _ in range(10):
            # 随机生成任务参数
            a, b, c = np.random.uniform(-2, 2, 3)
            objective, gradient = create_quadratic_task(a, b, c)

            # 随机初始化参数
            x = torch.tensor([np.random.uniform(-3, 3)], requires_grad=True)
            l2l_optimizer.reset_state()

            # 使用L2L优化器进行优化
            for step in range(10):
                loss = objective(x)
                loss.backward()

                with torch.no_grad():
                    # 使用L2L优化器计算更新
                    update = l2l_optimizer(x.grad.unsqueeze(0), x.unsqueeze(0))
                    x = x - update.squeeze()
                    x.requires_grad_(True)

            # 最终损失
            final_loss = objective(x)
            epoch_loss += final_loss.item()

        # 元更新
        avg_loss = epoch_loss / 10
        l2l_meta_optimizer.zero_grad()
        torch.tensor(avg_loss, requires_grad=True).backward()
        l2l_meta_optimizer.step()

        l2l_losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"   Epoch {epoch}: 平均损失 = {avg_loss:.4f}")

    # ==================== MAML 实验 ====================
    print("\n🔴 MAML 实验:")
    print("场景: 学习函数拟合的初始化")

    model = SimpleModel(input_dim=1, hidden_dim=32, output_dim=1)
    maml = MAML(model, inner_lr=0.01, meta_lr=0.001)

    # MAML元训练
    maml_losses = []
    for epoch in range(100):
        # 创建任务批次
        task_batch = []
        for _ in range(5):
            # 随机生成任务
            a, b, c = np.random.uniform(-2, 2, 3)

            # 生成任务数据
            x_task = torch.randn(10, 1)
            y_task = a * x_task ** 2 + b * x_task + c

            task_batch.append((x_task, y_task))

        # 元更新
        meta_loss = maml.meta_update(task_batch)
        maml_losses.append(meta_loss)

        if epoch % 20 == 0:
            print(f"   Epoch {epoch}: 元损失 = {meta_loss:.4f}")

    # ==================== 测试对比 ====================
    print("\n" + "=" * 60)
    print("测试阶段对比")
    print("=" * 60)

    # 创建测试任务
    test_a, test_b, test_c = 1.5, -1.0, 0.5
    test_objective, test_gradient = create_quadratic_task(test_a, test_b, test_c)

    # L2L测试
    print("\n🔵 L2L 测试结果:")
    x_l2l = torch.tensor([2.0], requires_grad=True)
    l2l_optimizer.reset_state()

    print(f"   初始: x = {x_l2l.item():.3f}, loss = {test_objective(x_l2l).item():.4f}")

    for step in range(10):
        loss = test_objective(x_l2l)
        loss.backward()

        with torch.no_grad():
            update = l2l_optimizer(x_l2l.grad.unsqueeze(0), x_l2l.unsqueeze(0))
            x_l2l = x_l2l - update.squeeze()
            x_l2l.requires_grad_(True)

    print(f"   最终: x = {x_l2l.item():.3f}, loss = {test_objective(x_l2l).item():.4f}")

    # MAML测试
    print("\n🔴 MAML 测试结果:")
    x_test = torch.randn(5, 1)
    y_test = test_a * x_test ** 2 + test_b * x_test + test_c

    # 快速适应
    fast_weights = maml.inner_loop(x_test, y_test, num_steps=10)

    # 测试适应后的性能
    test_pred = maml.functional_forward(x_test, fast_weights)
    test_loss = F.mse_loss(test_pred, y_test)

    print(f"   快速适应后测试损失: {test_loss.item():.4f}")

    return l2l_losses, maml_losses


# ==================== 关键区别总结 ====================
def summarize_differences():
    """
    总结L2L和MAML的关键区别
    """
    print("\n" + "=" * 80)
    print("🎯 L2L vs MAML 关键区别总结")
    print("=" * 80)

    print("\n📊 对比维度:")
    print("┌─────────────────┬─────────────────────┬─────────────────────┐")
    print("│   对比维度      │        L2L          │       MAML          │")
    print("├─────────────────┼─────────────────────┼─────────────────────┤")
    print("│ 学习目标        │ 学习优化器          │ 学习模型初始化      │")
    print("│ 应用场景        │ 优化问题            │ 模型适应            │")
    print("│ 输入            │ 梯度+参数+历史      │ 任务数据            │")
    print("│ 输出            │ 参数更新量          │ 模型预测            │")
    print("│ 替代对象        │ SGD/Adam等优化器    │ 随机初始化          │")
    print("│ 内循环          │ 优化步骤            │ 梯度下降            │")
    print("│ 外循环          │ 优化器参数更新      │ 初始化参数更新      │")
    print("└─────────────────┴─────────────────────┴─────────────────────┘")

    print("\n🔍 具体差异:")

    print("\n1️⃣ 学习内容不同:")
    print("   🔵 L2L: 学习 '如何优化' - 即优化算法本身")
    print("   🔴 MAML: 学习 '好的起点' - 即好的初始化参数")

    print("\n2️⃣ 应用方式不同:")
    print("   🔵 L2L: 替换传统优化器 (SGD → L2L)")
    print("   🔴 MAML: 提供好的初始化 (随机初始化 → MAML初始化)")

    print("\n3️⃣ 适用问题不同:")
    print("   🔵 L2L: 更适合你的参数优化问题")
    print("        - 学习theta, beta, phi的优化模式")
    print("        - 处理参数间的耦合关系")
    print("        - 自适应调整优化策略")

    print("\n   🔴 MAML: 更适合模型适应问题")
    print("        - 图像分类的few-shot学习")
    print("        - 新用户推荐系统适应")
    print("        - 多任务学习")

    print("\n4️⃣ 组合使用:")
    print("   💡 实际上可以组合使用:")
    print("      MAML提供好的初始化 + L2L提供好的优化器")
    print("      = 最强的元学习系统")


# ==================== 针对你的物理问题的建议 ====================
def recommendation_for_physics_problem():
    """
    针对物理参数优化问题的具体建议
    """
    print("\n" + "=" * 80)
    print("🎯 针对你的物理问题的建议")
    print("=" * 80)

    print("\n你的问题特点:")
    print("✓ 目标函数: 已设计好的物理目标函数")
    print("✓ 参数: theta, beta, phi (有物理意义)")
    print("✓ 需求: 更好的参数优化方法")

    print("\n💡 推荐方案:")
    print("1️⃣ 主要使用 L2L:")
    print("   - 学习如何优化你的物理参数")
    print("   - 处理参数间的物理耦合关系")
    print("   - 自适应不同的物理条件")

    print("\n2️⃣ 可选的组合方案:")
    print("   - L2L + 物理约束: 在L2L中嵌入物理约束")
    print("   - L2L + 贝叶斯优化: L2L处理局部优化，贝叶斯处理全局搜索")
    print("   - 多尺度L2L: 不同尺度的物理参数使用不同的L2L")

    print("\n3️⃣ 实现建议:")
    print("   - 收集历史优化数据进行元训练")
    print("   - 设计任务分布来模拟不同物理条件")
    print("   - 在L2L中融入物理先验知识")

    print("\n🚀 预期效果:")
    print("   - 更快的收敛速度")
    print("   - 更好的全局最优解")
    print("   - 自适应不同物理条件")
    print("   - 减少手动调参工作")


if __name__ == "__main__":
    # 运行对比实验
    l2l_losses, maml_losses = compare_l2l_vs_maml()

    # 总结区别
    summarize_differences()

    # 针对物理问题的建议
    recommendation_for_physics_problem()

    print("\n" + "=" * 80)
    print("🎉 总结: 对于你的物理参数优化问题，L2L是更好的选择！")
    print("=" * 80)