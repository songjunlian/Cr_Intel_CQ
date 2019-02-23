# 卷积神经网络和TensorFlow介绍

本教程对CNN和TensorFlow做了非常简洁的介绍，与当前常见的TensorFlow 和CNN教材不同，我们更多地侧重于TensorFlow的后台机制。本教程基于计算机来阐述，其可用于TensorFlow的定义，执行，储存等。同样，它也适用于其他任务。

**参考文献**
Tensorflow流程Demo: https://github.com/ringochuchudull/TensorFlowDemo
Tensorflow官方文献: https://tensorflow.google.cn/api_docs/python/tf/Graph

**本节内容安排如下:**

<!-- TOC -->

- [卷积神经网络和TensorFlow介绍](#%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8Ctensorflow%E4%BB%8B%E7%BB%8D)
  - [效果预期](#%E6%95%88%E6%9E%9C%E9%A2%84%E6%9C%9F)
  - [1. TensorFlow基础使用](#1-tensorflow%E5%9F%BA%E7%A1%80%E4%BD%BF%E7%94%A8)
    - [1.1 基本语法](#11-%E5%9F%BA%E6%9C%AC%E8%AF%AD%E6%B3%95)
      - [1.1.1 TensorFlow的基本单元Tensor](#111-tensorflow%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%8D%95%E5%85%83tensor)
      - [1.1.2 TensorFlow的算子operators](#112-tensorflow%E7%9A%84%E7%AE%97%E5%AD%90operators)
  - [2. TensorFlow中的计算图](#2-tensorflow%E4%B8%AD%E7%9A%84%E8%AE%A1%E7%AE%97%E5%9B%BE)
    - [2.1 获得计算图的内容](#21-%E8%8E%B7%E5%BE%97%E8%AE%A1%E7%AE%97%E5%9B%BE%E7%9A%84%E5%86%85%E5%AE%B9)
  - [3. 手写体检测CNN（Lenet-5）](#3-%E6%89%8B%E5%86%99%E4%BD%93%E6%A3%80%E6%B5%8Bcnnlenet-5)
  - [4. 练习（查看每层图像）](#4-%E7%BB%83%E4%B9%A0%E6%9F%A5%E7%9C%8B%E6%AF%8F%E5%B1%82%E5%9B%BE%E5%83%8F)
  - [5. FPGA板卡配置与登录](#5-fpga%E6%9D%BF%E5%8D%A1%E9%85%8D%E7%BD%AE%E4%B8%8E%E7%99%BB%E5%BD%95)
    - [5.1 硬件连接](#51-%E7%A1%AC%E4%BB%B6%E8%BF%9E%E6%8E%A5)
    - [5.2 网络设置与板卡登陆](#52-%E7%BD%91%E7%BB%9C%E8%AE%BE%E7%BD%AE%E4%B8%8E%E6%9D%BF%E5%8D%A1%E7%99%BB%E9%99%86)
      - [5.2.1 网络设置](#521-%E7%BD%91%E7%BB%9C%E8%AE%BE%E7%BD%AE)
    - [5.3 在不同系统中登录板卡](#53-%E5%9C%A8%E4%B8%8D%E5%90%8C%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%99%BB%E5%BD%95%E6%9D%BF%E5%8D%A1)
  - [附录：Lenet训练代码说明](#%E9%99%84%E5%BD%95lenet%E8%AE%AD%E7%BB%83%E4%BB%A3%E7%A0%81%E8%AF%B4%E6%98%8E)
    - [Loss Function](#loss-function)
    - [Optimizer](#optimizer)
    - [Accuracy](#accuracy)

<!-- /TOC -->


我们的教程针对的是TensorFlow的graph execution模式，与最近的TensorFlow版本引入的eager execution相比，eager execution可以在不首先构建计算图的情况下即时执行计算。

本教程仅介绍推演（inference）任务。

## 效果预期

1. 学会使用基础的TensorFlow。
2. 理解TensorFlow模型结构并学会从模型中提取信息。
3. 学会如何可视化CNN中间输出。

在以下教程中，我们将提供已构建的TensorFlow模型，请保存以便接下来的学习。

```python
#matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
```

## 1. TensorFlow基础使用
TensorFow是著名的机器学习构架，开发者能够轻松的在上面建立自己的模型。另一方面，TensorFow是嵌入在Python中的域特定语言（DSL）。作为域特定语言，TensorFlow提供了 primitives, 或者 APIs来构建计算机图用于机器学习模型。因此，在本教程中，我们采用TensorFlow作为编程语言：我们首先介绍TensorFlow中典型编程语言的元素，如语法，变量，数据类型，文字和运算符; 接下来，将展示如何执行TensorFlow构建的程序; 最后，会有一个关于使用TensorFlow的提示和技巧的简短列表

我们将在以下部分中研究TensorFlow中的计算图机制。

### 1.1 基本语法
当你安装好TensorFlow，我们还需要在python中导入如下代码（若没有安装好，请按照本教程[环境配置教程](https://github.com/corerain/CrTutorial/blob/master/Zh/00_Environment%20Configuration.md)进行安装）

```python
# import tensorflow and set its alias as "tf"
import tensorflow as tf
```
TensorFlow的基本操作逻辑如下图所示，简单来说Tensor即通过tf.constant, tf.Variable, tf.SparseTensor, tf.placeholder等创建出来的常量、变量、占位符等。而操作符（简称op）即可以理解为通过调用函数对张量进行处理的过程。
![avatar](https://i.loli.net/2019/01/08/5c3440dccc66d.png)
在 TensorFlow 中，所有在节点之间传递的数据都为 Tensor 对象(可以看作 n 维的数组)，常用图像数据的表示形式 为：`batch*height*width*channel`。接下来我们将会仔细介绍这些概念。

#### 1.1.1 TensorFlow的基本单元Tensor

在TensorFlow DSL中的可操作基单元是张量（tensor）（相关文档 [tf.Tensor](https://tensorflow.google.cn/api_docs/python/tf/Tensor)）， 简单来说，tensor是一个N维数组，和NumPy中的[ndarray](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.html)类似。下面的例子展示了如何用常数来构建tensor。

```python
tf.constant([[1.0, 2.0], [3.0, 4.0]]) # 创建2维2x2矩阵
```

除常数外，还可以创建变量tensor, 点击[document](https://tensorflow.google.cn/guide/variables)查看详情。可以通过get_variable创建变量的名称和大小，如下所示。Tensor名称在其name scope内只能是唯一的。

```python
tf.get_variable('a', [2, 2]) # 创建2维2x2张量Tensor
```


#### 1.1.2 TensorFlow的算子operators

我们可以利用TensorFlow里的算子[operators](https://tensorflow.google.cn/api_docs/python/tf/Operation)来构建计算。 tf.matmul 是一个直观的例子，它在两个2维tensors之间执行矩阵乘法。TensorFlow算子返回的对象是表征计算结果的tensor。

```python
tf.matmul(tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[1.0, 2.0], [3.0, 4.0]]))
```
输出：
```python
<tf.Tensor 'MatMul:0' shape=(2, 2) dtype=float32>
```


要运行计算，即获得tensor中的内容，我们需要使用tf.Session（参见 [here](https://tensorflow.google.cn/api_docs/python/tf/Session)）来初始化环境，以便来执行TensorFlow的程序。这部分不做深入探究。

注意**每个变量在开始前都需要初始化**，下面的示例使用random_normal_initializer来初始化tensor b的内容，使其为随机值且服从正态分布。

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.get_variable('b', [2, 2], initializer=tf.random_normal_initializer())
#创建一个2×2的随机矩阵，并将其初始化
c = tf.matmul(a, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
```
输出：
```python
[[-0.02581175  3.7516825 ]
 [-0.19293356  8.282628  ]]
```
TensorFlow官方仔细列出了所有可用的数学运算算子，用户可在此链接(https://www.tensorflow.org/api_guides/python/math_ops)中查看官方提供的算子。常用操作符包括：
```
tf.add(x, y) → 两个相同类型的张量相加，x + y
tf.subtract(x, y) → 两个相同类型的张量相减，x - y
tf.multiply(x, y) → 两个张量逐元素相乘
tf.pow(x, y) → 逐元素求幂
tf.exp(x) → 等价于 pow(e, x)，其中 e 是欧拉数（2.718…）
tf.sqrt(x) → 等价于 pow(x, 0.5)
tf.div(x, y) → 两个张量逐元素相除
tf.truediv(x, y) → 与 tf.div 相同，只是将参数强制转换为 float
tf.floordiv(x, y) → 与 truediv 相同，只是将结果舍入为整数
tf.mod(x, y) → 逐元素取余
```
## 2. TensorFlow中的计算图
大多数的计算机程序都可以用一张计算图来表示。由TensorFlow构建的程序明确地初始化了它的计算图。由TensorFlow构建的计算图是数据流图，其中每个节点表示计算（或操作），每个边是数据。下面的gif显示了如何为双层MLP（多层感知器）构建数据流图。在第一层，输入数据将首先通过矩阵乘法(tf.matmul)，偏向量加法(BiasAdd)和非线性激励(tf.nn.relu)。第二层在相同的设置下，处理除ReLU激励函数外的输出数据。权重和偏差向量是可变tensor。 注意，此图中还初始化了训练节点（梯度，更新）。

![图片](https://camo.githubusercontent.com/4ee55154486232ec9edd8f1a3bad4c4a146f6cfe/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74656e736f72735f666c6f77696e672e676966)


### 2.1 获得计算图的内容
本教程的目的主要是为了了解TensorFlow的机制以进一步进行优化和部署。这意味着有必要访问我们构建的图中的内容。假设我们构建的图形从输入到Softmax都与动画图形所示的计算完全相同。

```python
g = tf.Graph() # build a new graph
with g.as_default():
    # the second argument is the shape in this case 28x28x1 image
    input_tensor = tf.placeholder(tf.float32, [28, 28, 1])
    
    # reshape it as a column vector
    x = tf.reshape(input_tensor, [1, 784], name='x')
    
    # The first perceptron layer
    W1 = tf.get_variable('W1', [784, 1024])
    b1 = tf.get_variable('b1', [1024])
    
    # Perform the matrix multiply and add bias
    y1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W1), b1), name='y1')
    
    # The second perceptron layer
    W2 = tf.get_variable('W2', [1024, 10])
    b2 = tf.get_variable('b2', [10])
    y2 = tf.nn.bias_add(tf.matmul(y1, W2), b2, name='y2')
    
    logits = tf.nn.softmax(y2, name='logits') # softmax output is named as "logits"
```

TensorFlow把所有与图相关的信息都放在[tf.Graph class](https://tensorflow.google.cn/api_docs/python/tf/Graph)中，就像上例中的`g`一样。该图形对象将图形的定义保存为GraphDef，可以通过as_graph_def()访问。你可以通过其node属性遍历GraphDef中的每个节点。每个节点都有名称，op用于操作名称，input用于输入到当前节点的节点名称列表，以及其他可访问属性。在下面的例子中，我们输出每个MatMul节点的属性。定义节点的语法是一个[Protocol Buffer](https://developers.google.com/protocol-buffers/)，可以查阅其文档以获得更多细节。

```python
graph_def = g.as_graph_def()
for node in graph_def.node:
    if node.op == 'MatMul':
        print(node)
```
输出：
```python
name: "MatMul"
op: "MatMul"
input: "x"
input: "W1/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "transpose_a"
  value {
    b: false
  }
}
attr {
  key: "transpose_b"
  value {
    b: false
  }
}

name: "MatMul_1"
op: "MatMul"
input: "y1"
input: "W2/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "transpose_a"
  value {
    b: false
  }
}
attr {
  key: "transpose_b"
  value {
    b: false
  }
}
```

除了访问节点定义，我们还可以通过tf.Graph的`get_tensor_by_namemethod`方法来读取每个图中每一节点的内容。这个方法会返回一个Tensor对象，可以通过sess.run读取其中的内容。请注意：tensor的名字并不完全是对应节点的名字，我们需要附加上device placement。 在后面的例子中，权重W1被放到index0的设备上。

```python
with tf.Session(graph=g) as sess: # we need to explicitly set the graph or the default graph will be used.
    sess.run(tf.global_variables_initializer())
    
    # read the content of initialised W1
    W1_tensor = g.get_tensor_by_name('W1:0')
    print(sess.run(W1_tensor))
```

输出：
```
[[-0.04166514 -0.04794802 -0.01143964 ... -0.0312358   0.05538596
   0.01754197]
 [ 0.05225373  0.03787024  0.00387241 ... -0.03806394  0.01877896
   0.02079131]
 [-0.03578468  0.02372336 -0.05597354 ...  0.000456    0.03292146
   0.04967514]
 ...
 [ 0.03961485  0.00088048 -0.05151795 ...  0.01873089  0.02834749
   0.03316635]
 [-0.05604851  0.02302834 -0.01317953 ... -0.054464   -0.03014182
   0.04097571]
 [-0.00646909 -0.05257693 -0.01661416 ... -0.02089745  0.05614159
   0.00930553]]
```

要读取于Input节点的tensor内容，我们需要在会话执行时向其提供内容。可以查阅资料https://tensorflow.google.cn/api_docs/python/tf/placeholder, 如果还是不能完全明白TensorFlow，可以看这个网址：https://github.com/ringochuchudull/TensorFlowDemo。
这是一个小的展示，在一张简洁的海报中总结了TensorFlow 以及 NumPy的基础知识。

```python
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    # read the final classification result of a random image
    logits_tensor = g.get_tensor_by_name('logits:0')
    print(sess.run(logits_tensor, feed_dict={input_tensor: np.random.random((28, 28, 1))}))
```

```python
[[0.03610415 0.06912688 0.07223938 0.16913089 0.0772726  0.1951995
  0.07214345 0.18331014 0.04666999 0.07880298]]
```

## 3. 手写体检测CNN（Lenet-5）
基于前面的讨论，我们可以构建卷积神经网络（CNN）并在TensorFlow中读取其结构和系数。CNN可以看作是包含卷积层的特殊计算图，在TensorFlow中是`tf.nn.conv2d`。(https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d) 可以在本教程中访问卷积层原理的详细信息。

在传统的CNN网络中，每层的输入和输出都是特征图，通常是3-Dtensor(批量输入时为4-D)。 一幅特征图可以被看成一个多通道图像，它能够代表自然图像或者从潜在的空间里提取的特征。每个特征图的形状是空间特征图的高度和宽度，也是图像通道的数量。卷积层在每个空间特征映射中执行2D卷积，并将结果汇总在一起用于不同的输出通道。除了卷积层之外，最大池层对于减小特征映射的大小以提取更高级别的特征也同样重要。

我们在TensorFlow中构建一个CNN的实例，其用于手写体数字的识别。这个CNN的架构被称为[LeNet](http://yann.lecun.com/exdb/lenet/)。它需要使用MNIST数据集进行训练。类似的教程可以在以下链接找到：(https://tensorflow.google.cn/tutorials/deep_cnn)

创建python文件`Tutorial01.py`，并且输入以下代码：

```python
def lenet(images, keep_prob):
    """
    Args:
        images: a 4-D tensor that holds batched input images
    Return:
        A tensor that contains classification probabilities result, and a dictionary
        of all intermediate tensors.
    """    
    end_points = {}
    # Input shape of 28,28,1 and -1 is just for TF purposes
    end_points['images'] = tf.reshape(images, [-1, 28, 28, 1])
    
    # Define the scope
    with tf.variable_scope('conv1'):
        # Define the weights for the convolution aka. Kernel size, kernel size, stride and number of filters
        w1 = tf.get_variable('weights', [5, 5, 1, 32])
        
        # Define the bias
        b1 = tf.get_variable('biases', [32],
                             initializer=tf.zeros_initializer())
        # Perform the computation and apply ReLU function
        # First do the conv2d with weights w1 in the SAME namespace then add the bias, later activation function
        end_points['conv1'] = tf.nn.relu(
            tf.nn.conv2d(end_points['images'], w1, [1, 1, 1, 1], 'SAME') + b1)
    
    # Add a max-pooling operation with kernel 2x2 and stride 1
    end_points['pool1'] = tf.nn.max_pool(
        end_points['conv1'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    with tf.variable_scope('conv2'):
        w2 = tf.get_variable('weights', [5, 5, 32, 64])
        b2 = tf.get_variable('biases', [64],
                             initializer=tf.zeros_initializer())
        end_points['conv2'] = tf.nn.relu(
            tf.nn.conv2d(end_points['pool1'], w2, [1, 1, 1, 1], 'SAME') + b2)
    end_points['pool2'] = tf.nn.max_pool(
        end_points['conv2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    end_points['flatten'] = tf.reshape(end_points['pool2'], [-1, 7 * 7 * 64])
    with tf.variable_scope('fc3'):
        w3 = tf.get_variable('weights', [7 * 7 * 64, 1024])
        b3 = tf.get_variable('biases', [1024],
                             initializer=tf.zeros_initializer())
        end_points['fc3'] = tf.nn.relu(tf.matmul(end_points['flatten'], w3) + b3)
        
    end_points['dropout'] = tf.nn.dropout(end_points['fc3'], keep_prob)
    with tf.variable_scope('fc4'):
        w4 = tf.get_variable('weights', [1024, 10])
        b4 = tf.get_variable('biases', [10],
                             initializer=tf.zeros_initializer())
        end_points['fc4'] = tf.matmul(end_points['fc3'], w4) + b4
    
    return end_points['fc4'], end_points
```
然后，我们基于MNIST数据集训练了这个CNN网络（参考）http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html)


**注意**：
如果无法下载MNIST训练集和测试集数据，请做如下改动mnist.py改动37-40行，主要是调整数据集下载的服务器地址。也可在提供的github中直接下载数据集。

```python
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
# DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
DEFAULT_SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
```

以下是我们用于训练模型的代码段：

```python
# NOTE: You don't need to run this code snippet since we have already trained it
# and it will consume lots of resources on our server.

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

g = tf.Graph()
with g.as_default():
    images = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    logits, end_points = lenet(images, keep_prob)
    
    # Nodes for training
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    train = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(20000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            _, loss_val = sess.run([train, loss],
                                   feed_dict={images: batch_xs,
                                              labels: batch_ys,
                                              keep_prob: 0.5})
        
            if i % 100 == 0:
                print('Loss value of a training batch at step %5d: %f' % (i, np.mean(loss_val)))
            if i % 1000 == 0:
                acc = sess.run(accuracy,
                               feed_dict={images: mnist.test.images,
                                          labels: mnist.test.labels,
                                          keep_prob: 1.0})
                print('Accuracy after running %5d steps: %f' % (i, acc))
        
        # save the trained model
        saver.save(sess, "./mnist_lenet_log/")
```

运行`turial01.py`输出结果：

```shell
Accuracy after running     0 steps: 0.087900
Loss value of a training batch at step   100: 2.307177
Loss value of a training batch at step   200: 2.291096
Loss value of a training batch at step   300: 2.272854
Loss value of a training batch at step   400: 2.267840
Loss value of a training batch at step   500: 2.275491
```



## 4. 练习（查看每层图像）
在本练习中，建议你在每个卷积层和最大池化层之后可视化中间特征图，你可以观察一张输入数字图片是如何转换为特征值。我们提供以下代码片段的基本构架，你需要完成下面的任务：

正确识别lenet函数中所有卷积层和最大池化层的end_point tensor。 注意，看到多个划线的地方来寻找网络。 这些tensor可以视为这些层的输出。
根据输入测试图像收集这些tensor的值。 可以从测试数据集（mnist.test.images）中选择一个图像。对于每个tensor值，你可以获得tensor任何通道的2D图像。 提示：tensor值是NumPy ndarray。通过plt.imshow可视化2D图像。


```python
# EXERCISE VERSION
def visualize_tensor(image, key, channel_idx, axis):
    """
    Visualize a tensor in the trained LeNet model.
    Args:
        image: a test image
        key: the key to the tensor in end_points
        channel_idx: index of the channel to be visualized
        axis: a pyplot Axis object
    """
    saver = tf.train.Saver()

    with g.as_default():
        images = tf.placeholder(tf.float32, shape=[None, 784])
        labels = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32)
        logits, end_points = lenet(images, keep_prob)

        # Nodes for training
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        train = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)

        # accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # ensure the folder path is correct
        with tf.Session(graph=g) as sess:
            saver.restore(sess, 'mnist_lenet_log/') 
            
            # TODO: finish the line to get the tensor value of end_points[key]
            tensor_val = sess.run("""BLANK""", feed_dict={images: [image], keep_prob: 1.0})
            
            # TODO: get the 2D image at channel "channel_idx"
            image_2d = tensor_val[0, """BLANK"""]
            
            # TODO: visualize
            axis.set_title(key)
            axis.imshow("""BLANK""", cmap='gray')
```
```python
# TODO: use visualize_tensor to visualize the channel 0 of all convolutional layers and max-pooling layers of the first test image in MNIST
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8, 6))
```

## 5. FPGA板卡配置与登录

在完成了开发环境和部署环境的安装之后，我们可以进行FPGA板卡功能的验证。
此章与上文两章的逻辑关系为：用户在TensorFlow开发环境中完成算法模型的开发，在部署环境中使用Rainbuilder对模型进行8bit量化并输出FPGA板卡所能识别的硬件描述文件，最终在板卡上实现模型的运行。
此章节所展现的是最终的板卡验证结果。

### 5.1 硬件连接
1. 将雨人v3.1加速器板放在平坦且不导电的表面上。
2. 将雨人加速器板上的以太网端口直接连接到主机PC的以太网端口。
3. 通过USB Type-C接口为加速器板供电。我们建议使用5V 2A手机充电器为电路板供电。 PC或笔记本电脑的USB端口可能无法提供所需的足够和稳定的电量。
4. 然后加速器板正常供电，板上（SD卡附近）的绿色LED亮起。如果使用带风扇进行散热的版本，正上电之后，风扇开启。
![board.PNG](https://i.loli.net/2019/02/14/5c65352dbaf79.png)

### 5.2 网络设置与板卡登陆
#### 5.2.1 网络设置
在主机系统中，**必须手动配置以太网到加速器板的网络接口IPv4地址，否则无法链接雨人V3板卡**，配置参数如下：

|设置|值|
|:---: |:---:|
|本地主机IPv4 | 192.168.123.10|
|网络掩码| 255.255.255.0|
|雨人V3板卡IP地址| 192.168.123.8|

linux系统设置方法如图

![avatar](https://i.loli.net/2018/10/04/5bb593abd80d6.png)

Windows 10系统设置方法如图

![avatar](https://i.loli.net/2018/10/04/5bb5a645b5e97.png)

### 5.3 在不同系统中登录板卡
**在Linux系统中登录板卡**
在配置好主机系统的网络参数后，在Linux中打开命令行，使用以下命令登录板卡：

```shell
ssh root@192.168.123.8
```
输入密码：
```shell
letmein
```
- **在虚拟机中登录板卡**
如本章2.1小节配置好网络后，打开虚拟机的网络设置，将网卡的连接方式从网络地址转换（NAT）更改为**桥接网卡**。
![choose ethernet card.PNG](https://i.loli.net/2019/02/14/5c65343923d11.png)

在配置好网络参数之后，接下来的操作同Linux系统中登录板卡一般，通过``ssh``指令，便能通过网口与板卡进行连接。


**在Windows 10系统中登录板卡**
如果需要在Windows系统下登陆雨人V3板卡，建议下载MobaXterm，该软件可以支持远程脚本编辑，图片查看等，下载链接：https://mobaxterm.mobatek.net/
![avatar](https://i.loli.net/2018/10/04/5bb5a7565d2ea.png)

同理，执行`ssh root@192.168.123.8`，并输入密码：`letmein`，即可在Windows 10上完成板卡的登录。



**常见问题：**
1. 路径问题



2. 模型文件导入问题
当出现以下错误时，请检查模型文件路径下是否保存了TensorFlow模型文件, *.index, *.data-00xxxx, *.data。
```shell
2019-02-17 00:43:40.346806: W tensorflow/core/framework/op_kernel.cc:1318] OP_REQUIRES failed at save_restore_tensor.cc:170 : Data loss: Unable to open table file .\mnist_lenet_log: Unknown: NewRandomAccessFile failed to Create/Open: .\mnist_lenet_log : 拒绝访问。
; Input/output error
```

## 附录：Lenet训练代码说明
### Loss Function
```python
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
```
上面的公式定义了模型训练的loss function，这里采用了分类问题中常用的cross-entropy loss，其中labels指的是训练样本的ground-truth标签，logits是网络当前的预测值。具体的计算方法如下：
1. 首先使用softmax函数将logits归一化
   
   $softmax(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j )}$
2. 计算Cross-entropy loss
   
   $H(y',y)=-\sum_i y'_i \cdot log(y_i)$, 其中$y'_i=1$如果$y_i$的标签为第$i$类。
3. 计算平均值，输出loss最终值。

### Optimizer
```python
train = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
```
使用Adadelta做为优化器，tensorflow提供了多种优化器，包括`adadelta`,`adagrad`,`adam`,`ftrl`,`momentum`,`sgd`和`rmsprop`。不同优化器的网络参数更新方式和学习率调整策略有所不同。Adadelta优化器的具体设计请参考[Adadelta](https://arxiv.org/abs/1212.5701).

### Accuracy
```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
在网络的最后输出中，找出最大项，最大项对应的类别即为最终的预测类别。
如果预测类别与ground-truth标签相等，则计为正确。在整个数据集上求平均，可得到总体的准确率。