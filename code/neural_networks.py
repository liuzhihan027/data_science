# -*- coding: utf-8 -*-

from __future__ import division
from collections import Counter
from functools import partial
from linear_algebra import dot
import math, random
import matplotlib
import matplotlib.pyplot as plt


# 简单神经元
def step_function(x):
    return 1 if x >= 0 else 0


# 输入权重,偏移量,和参数,返回神经元输出
def perceptron_output(weights, bias, x):
    """returns 1 if the perceptron 'fires', 0 if not"""
    return step_function(dot(weights, x) + bias)


# 一个类似逻辑回归的函数,平滑函数
def sigmoid(t):
    return 1 / (1 + math.exp(-t))


# 神经元的输出
def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))


# 前馈神经网络
def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""

    outputs = []

    # 遍历每层
    for layer in neural_network:
        input_with_bias = input_vector + [1]  # 添加偏倚

        output = [neuron_output(neuron, input_with_bias)  # compute the output
                  for neuron in layer]  # for this layer

        outputs.append(output)  # and remember it

        # the input to the next layer is the output of this one
        # 本层的输出作为下一层的输入
        input_vector = output

    return outputs


# 反向传播(网络,输入值,目标值)
def backpropagate(network, input_vector, target):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    # 类似逻辑回归的导数计算
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # adjust weights for output layer (network[-1])
    # 调整输出层的权重
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    # 向隐藏层反向传播误差
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer (network[0])
    # 调整隐藏层的权重（网络〔0〕）
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input


def patch(x, y, hatch, color):
    """return a matplotlib 'patch' object with the specified
    location, crosshatch pattern, and color"""
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)


def show_weights(neuron_idx):
    weights = network[0][neuron_idx]
    abs_weights = map(abs, weights)

    grid = [abs_weights[row:(row + 5)]  # turn the weights into a 5x5 grid
            for row in range(0, 25, 5)]  # [weights[0:5], ..., weights[20:25]]

    ax = plt.gca()  # to use hatching, we'll need the axis

    ax.imshow(grid,  # here same as plt.imshow
              cmap=matplotlib.cm.binary,  # use white-black color scale
              interpolation='none')  # plot blocks as blocks

    # cross-hatch the negative weights
    for i in range(5):  # row
        for j in range(5):  # column
            if weights[5 * i + j] < 0:  # row i, column j = weights[5*i + j]
                # add black and white hatches, so visible whether dark or light
                ax.add_patch(patch(j, i, '/', "white"))
                ax.add_patch(patch(j, i, '\\', "black"))
    plt.show()




# 算法应用
if __name__ == "__main__":

    # 与或非异或测试
    xor_network = [
        [[20, 20, -30],  # AND
         [20, 20, -10]],  # OR
        [[-60, 60, -30]]  # 异或
    ]

    for x in [0, 1]:
        for y in [0, 1]:
            print x, y, feed_forward(xor_network, [x, y])

    # 数字识别
    raw_digits = [
        """11111
           1...1
           1...1
           1...1
           11111""",

        """..1..
           ..1..
           ..1..
           ..1..
           ..1..""",

        """11111
           ....1
           11111
           1....
           11111""",

        """11111
          ....1
          11111
          ....1
          11111""",

        """1...1
          1...1
          11111
          ....1
          ....1""",

        """11111
          1....
          11111
          ....1
          11111""",

        """11111
          1....
          11111
          1...1
          11111""",

        """11111
          ....1
          ....1
          ....1
          ....1""",

        """11111
          1...1
          11111
          1...1
          11111""",

        """11111
          1...1
          11111
          ....1
          11111"""]


    # 将数据扁平化成数组
    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]


    inputs = map(make_digit, raw_digits)

    # 定义结果,类似对角线矩阵
    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(0)  # to get repeatable results
    input_size = 25  # each input is a vector of length 25
    num_hidden = 5  # we'll have 5 neurons in the hidden layer(每层五个神经元)
    output_size = 10  # we need 10 outputs for each input(每层输出10个结果)

    # each hidden neuron has one weight per input, plus a bias weight
    # 初始化随机隐藏层
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # each output neuron has one weight per hidden neuron, plus a bias weight
    # 初始化每层输出,随机生成
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    # the network starts out with random weights
    # 初始化网络,随机隐藏层,随机输出层
    network = [hidden_layer, output_layer]

    # 10,000 iterations seems enough to converge
    # 10000次的迭代使其收敛
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)


    def predict(input):
        return feed_forward(network, input)[-1]


    for i, input in enumerate(inputs):
        outputs = predict(input)
        print i, [round(p, 2) for p in outputs]

    print """.@@@.
...@@
..@@.
...@@
.@@@."""
    print [round(x, 2) for x in
           predict([0, 1, 1, 1, 0,  # .@@@.
                    0, 0, 0, 1, 1,  # ...@@
                    0, 0, 1, 1, 0,  # ..@@.
                    0, 0, 0, 1, 1,  # ...@@
                    0, 1, 1, 1, 0])  # .@@@.
           ]
    print

    print """.@@@.
@..@@
.@@@.
@..@@
.@@@."""
    print [round(x, 2) for x in
           predict([0, 1, 1, 1, 0,  # .@@@.
                    1, 0, 0, 1, 1,  # @..@@
                    0, 1, 1, 1, 0,  # .@@@.
                    1, 0, 0, 1, 1,  # @..@@
                    0, 1, 1, 1, 0])  # .@@@.
           ]
    print

    for i in range(4):
        show_weights(i)
