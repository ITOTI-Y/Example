
# BP神经网络

## 简介

`BP_neural_utils.py` 文件包括一个名为 `bp_neural` 的类，用于实现一个简单的BP（反向传播）神经网络。该类支持前向传播、反向传播、参数更新、预测、准确率计算、混淆矩阵绘制等功能。

## 依赖

- numpy
- seaborn
- matplotlib
- sklearn.metrics

## 类结构

### bp_neural

该类包括以下方法：

- `__init__(self, train_imgs, train_labels, n_hidden=100, method='softmax')`：初始化神经网络，设置隐藏层节点数和激活函数等。
- 前向传播方法。
- 反向传播方法。
- 参数更新方法。
- 预测方法。
- 准确率计算方法。
- 混淆矩阵绘制方法。

## 示例

以下是如何使用该类的示例：

```python
from BP_neural_utils import bp_neural

# 创建一个BP神经网络实例
neural_network = bp_neural(train_imgs, train_labels, n_hidden=100)

# 训练神经网络
neural_network.front_prop()
neural_network.back_prop()
neural_network.update_params()

# 预测新数据
neural_network.predict(test_imgs)

# 查看准确率
neural_network.accuracy()

# 绘制混淆矩阵
neural_network.plot()
```

## 注意事项

确保输入数据是适当的形状，并且已经进行了适当的预处理（例如归一化或标准化）。

