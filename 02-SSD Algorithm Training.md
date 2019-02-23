# 基于SSD框架的目标检测卷积神经网络设计

本课程我们介绍用于工程界的目标检测场景的SSD算法架构。从本章节可以了解到SSD神经网络的训练基本步骤；如何基于VGG建立特征提取网络；前后处理的基本概念和功能；从模型文件中恢复网络模型并进行测试等。

**本节内容安排如下**：

- [1. SSD目标检测框架概述](#1-ssd目标检测框架概述)
- [2. 测试数据集结构与TFrecoard格式转换](#2-测试数据集结构与tfrecord格式转换)
    - [2.1 原理介绍](#21-原理介绍)
    - [2.1 操作流程](#21-操作流程)
        - [2.1.1 在IDE界面中进行操作](#211-在ide界面中进行操作)
        - [2.1.2 在终端中进行操作](#212-在终端中进行操作)
- [3. 如何构建SSD网络的图像预处理](#3-如何构建ssd网络的图像预处理)
- [4. 基于VGG构建的SSD神经网络结构](#4-基于vgg构建的ssd神经网络结构)
    - [4.1 使用Tensorflow建立SSD神经网络架构](#41-使用tensorflow建立ssd神经网络架构)
    - [4.2 自定义类VGG神经网络](#42-自定义类vgg神经网络)
- [5. 进行神经网络训练](#5-进行神经网络训练)
    - [5.1 原理介绍](#51-原理介绍)
    - [5.2 操作流程](#52-操作流程)
        - [5.2.1 在IDE界面中进行操作](#521-在ide界面中进行操作)
        - [5.2.2 在终端界面中进行操作](#522-在终端界面中进行操作)
- [6. 进行图片测试验证结果](#6-进行图片测试验证结果)
    - [6.1 原理介绍](#61-原理介绍)
    - [6.2 操作流程](#62-操作流程)
        - [6.2.1 在IDE界面中进行操作](#621-在ide界面中进行操作)
        - [6.2.2 在终端界面中进行操作](#622-在终端界面中进行操作)
- [7. cv2模块加载失败的解决方案](#7-cv2模块加载失败的解决方案)
- [8. 练习](#8-练习)


本教程使用文件夹`Tutorial2_codes`中的代码。

## 1. SSD目标检测框架概述
深度学习算法在目标检测领域已有比较成熟的应用，主要框架分为One-Shot Detector和Two-Shot Detector两种。One-Shot，顾名思义，指的是只运行一遍算法，便计算边界框（bounding box）中物体的置信度。One-Shot相对于Two-Shot在基本保持相同精度的情况下，速度大幅提升，更加实用于工业级应用。

本教程中从采用经典的Single-Shot Multibox Detector（SSD），具体原理可以参考:https://arxiv.org/abs/1512.02325。

SSD网络主要由两部分组成：主干网络（特征提取网络）、多尺度上的检测分支。
![ssd.jpg](https://i.imgur.com/Zu6nw3A.png)
最初SSD的特征提取网络使用了VGG-16神经网络，关于VGG-16的网络结构请参考[原论文链接](https://arxiv.org/pdf/1409.1556.pdf)。VGG网络通过嵌套卷积层（CONV）+池化层（Pooling）完成特征提取，随着网络层数的加深，特征图尺寸越来越小。不同层次的特征图能代表不同层次的语义信息，低层次的特征图能代表低层语义信息(含有更多的细节)，适合小尺度目标的学习。高层次的特征图能代表高层语义信息，适合对大尺度的目标进行深入学习。

SSD包含连接在不同尺度的特征图上的检测分支，每个检测分支负责预测该尺度上的Anchor的类别classification和定位localization信息。基本可以将Anchor视为一些基准检测框，而SSD的目的是如何对于基准框进行微调来计算最终的高精度检测结果。

关于SSD中Anchors的概念可以参考下图：
![example.PNG](https://i.loli.net/2018/11/29/5bff52b8e1cd7.png)
上图a中我们希望使用SSD检测图中的猫（蓝框）和狗（红框），其中不同颜色代表我们上面提到的类别。由于猫和狗在图中的尺寸不同，在SSD中需要使用不同的分支完成对于猫和狗的检测。b图代表8x8的特征图对应的分支，c图代表4x4的特征图分支。具体来说，b层位于浅层，包含更多细节信息，因此对应于猫的检测，c层位于深层，对应于狗的检测。b、c图中的虚线框代表Anchor基准框，基准框包含不同的尺度和长宽比，以保证对于各种尺寸和形状物体的覆盖。如c图所示，SSD通过计算每个anchor负责区域的位置微调$\Delta(cx, cy, w, h)$和对应的类别置信度$(c_1, c_2, ..., c_p)$，最终实现对于物体的检测。


## 2. 测试数据集结构与TFrecoard格式转换
### 2.1 原理介绍

神经网络能够正常检测目标的基础是基于大量的数据的训练。出于实验目的，本次只使用20张图片作为训练数据集。对于数据量较小而言，一般选择直接将数据加载进内存，然后再分batch输入网络进行训练。如果训练集太大，这种方法过于消耗内存。如果使用Tensorflow进行训练时，此时推荐使用[TFrecord](https://tensorflow.google.cn/tutorials/load_data/tf-records)数据格式，该格式的——TFRecords内部使用了“Protocol Buffer”二进制数据编码方案，在tensorflow上易于进行大量数据的输入。

在本次实验中，我们标定数据集使用的格式：
>图像名  图像中物体数量  物体1:物体1对应的label_id  2(预定义字段)  左上角x坐标  y坐标 右下角x坐标 y坐标  物体2...

**在数据集文件中包含：**
```shell
1530697831_1.jpg 1 face:1 2 414.0 207.0 536.0 304.0
1530697831_2.jpg 1 face:1 2 2461.224 985.714 2981.633 1695.918
1530697831_3.jpg 1 face:1 2 3746.939 1224.49 4573.469 1897.959
1530697831_4.jpg 1 face:1 2 155.0 113.0 282.0 278.0
1530697831_5.jpg 1 face:1 2 1910.204 1518.367 3618.367 3336.735
1530697831_6.jpg 1 face:1 2 2442.857 1126.531 3704.082 2014.286
1530697831_7.jpg 1 face:1 2 2473.469 1028.571 3936.735 2387.755
```

这种数据集格式易读，直观显示图片名称，目标物体种类，在对应图片中目标的位置。根据这个格式，用户可以自由的制作自己的人脸图片数据集进行训练。但是tensforflow并不接受这种格式作为标准输入，需要使用TFrecord格式。

第一步是将可读的数据集转换为二进制TFRecord文件。
本次实验提供`tf_convert_data.py`将上方的数据格式转换为`TFRecord`。

使用`python tf_convert_data.py`将可读数据集转换为`TFRecord`。在运行函数时，需要提供图片路径，输出TFRecord文件名称和输出文件路径。

```
python tf_convert_data.py --dataset_name=test_release --dataset_dir=../imagetxt/ --output_dir=./ --shuffle=True
```

### 2.1 操作流程

本教程中提供了两种格式转换方法，在IDE界面中进行操作和在终端界面中进行操作。

**注意**：需要对示例提供的代码中数据集的路径进行修改，将其指向你实际的路径。

#### 2.1.1 在IDE界面中进行操作

打开一个新终端，在终端中输入`anaconda-navigator`后，便可启动Anaconda。在Anaconda中将环境切换为TensorFlow后，点击Launch启动VSCode进行操作。

具体流程可参见教程1。

在VSCode中打开`tf_convert_data.py`文件，在运行前需要确认VSCode的Python插件已被激活。

![avatar](https://i.loli.net/2019/01/25/5c4a85f463976.jpg)

之后进行Python解析器的选择和Python环境路径的确认，具体流程可参见教程1。

对第36行的`data_dir`进行修改，指向标记文件所在的路径中。
![avatar](https://i.loli.net/2019/01/25/5c4add3c0a8bf.png)

确认无误后，可单机右键，选择在终端中运行Python文件，即可运行格式转化程序。



#### 2.1.2 在终端中进行操作

打开终端，输入`source activate 环境名称`，其中环境名称指的是在Anaconda中创建的TensorFlow换环境名称，可以回顾教程1的1.1.2节，在示例中环境名称为Tensorflow。

在输入命令之后，可以发现用户名前面出现了TensorFlow的环境名称，说明Tensorflow的环境已被成功开启，示意如下。
```shell
username@username-VirtualBox:~$source activate Tensorflow
(Tensorflow)username@username-VirtualBox:~$ 
```

之后，在Tutorial2_codes下目录中运行如下代码：

```shell
sh script/1_run_convert.sh
```

即可执行脚本，进行数据格式的转换。可以通过`cat script/1_run_conver.sh`来浏览脚本中的代码，脚本内容如下。
```shell
python tf_convert_data.py \ #调用py文件
       --dataset_name=test_release \ #数据库名称
       --dataset_dir=../imagetxt/ \ #输入路径
       --output_dir=./ \ #输出路径
       --shuffle=True
```
可见，这两种方法在本质上是一致的。


## 3. 如何构建SSD网络的图像预处理

图像预处理是深度学习算法中非常重要的步骤。有些预处理方法可以帮助排除数据库中一些干扰因素，有些可以让深度学习网络更好地收敛，也有些是为了满足算法框架本身的要求。常见的图像预处理方式有简单缩放、均值消减、归一化、白化等。

在本次试验中，我们采用OpenCV实现预处理。OpenCV是一个开元发行的跨平台计算机视觉库，它由一系列C函数和少量C++类构成，同时提供了Python、Ruby、MATLAB等语言的接口，帮助实现了图像处理和计算机视觉方面的很多通用算法。

开始前，我们先简单介绍以下图像的基本知识。计算机中的图像是由像素组成的矩阵表示的。以一张1080p的图像为例，图像中包含1920x1080x3个像素点，每个像素点的值为一个范围在[0，255]的数字，其中3代表通道数，自然图像是由红（R）、绿（G）、蓝（B）三个通道组成的。但是这种原始表达形式的数据量很大，不利于传输和保存，通常来说，图像数据都是经过编码压缩来存储的，其中常见的编码格式包括jpeg、bmp、png等。
然而深度学习网络一般以原始像素格式表达的图像作为输入，因此需要将编码过的图像解码导入内存，再进行处理。


>**前处理具体步骤如下：**
>   1. 读取图像 --> 使用cv2.imread()
>   2. 调整通道顺序 --> BGR转RGB，使用cv2.cvtColor()
>   3. 图像缩放到256x256 --> 使用cv2.resize()
>   4. 图像消减mean --> 参考numpy示例，R通道-123，G通道-117， B通道-104 
其中步骤4主要是为了消除亮度对于网络训练的影响

我们会在后面第7步骤练习部分进行预处理练习。

## 4. 基于VGG构建的SSD神经网络结构
本步骤搭建SSD神经网络，根据论文可知，SSD的特征提取和分类网络如下图所示：
![SSD.PNG](https://i.loli.net/2018/11/29/5bff4afe3004f.png)

该网络的特征提取部分使用VGG-16，主要使用网络使用卷积与最大池化操作。需要了解卷积与池化的概念请参考下图：

通过卷积，可以提取图片矩阵中的特征值。卷积的工作原理请参考本事例，事例使用输入5 x 5的图片矩阵，滤波器使用3 x 3矩阵，计算过程如下图所示：

<img src="https://raw.githubusercontent.com/stdcoutzyx/Paper_Read/master/blogs/imgs/6.gif" width=400 height=256 />

最大池化的概念是在原始图片上取n x n得矩阵，然后在n x n矩阵寻找最大值作为输出，放置在输出矩阵中。如下图所示：

<img src="https://i.loli.net/2018/11/30/5c0158ce50531.png" width=700 height=256 />

### 4.1 使用Tensorflow建立SSD神经网络架构
SSD架构使用VGG16构建第一层到到第六层的特征提取网络，根据VGG16结构，我们定义如下网络结构：
![VGG.PNG](https://i.loli.net/2018/11/29/5bff4b9dad933.png)

本事例使用了Tensorflow的高层API [Tensorflow.slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)，它使构建网络变得更加简单，让算法工程师更加专注于网络结构。比如使用`slim.repeat`函数可以快速创建多层结构相同的卷积滤波器，而不用每次层卷积都添加一次代码。由于VGG大量使用了多层结构相同的卷积和池化操作，使用`slim.repeat`会很方便。

下列代码展示在Tensorflow下定义SSD神经网络，Tensorflow构建的网络结构对应上方图片中高亮部分的网络结构。
```python
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
```

### 4.2 自定义类VGG神经网络
接下来我们进入`net`文件夹，打开`ssd_KYnet_v3_5b`定义`ssd_net`来构建自己的模型，从第439行开始自定义`ssd_net`函数：

```python
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            net_scale=1.0,
            scope='ssd_KYnet_v2'):
    # SSD net definition.
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
```

使用类VGG网络搭建更小的神经网络进行人脸目标检测，我们定义一个参数`net_scale`用来量化通道数的大小。使用卷积 + 池化的结构。 1层32通道3x3卷积 => 2x2最大池化 => 2层32通道3x3卷积 => 2x2最大池化 => 3层48通道3x3卷积 => 2x2最大池化 => 2层64通道3x3卷积 => 1层96通道3x3卷积。

请重点参考[Tensorflow.slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)中`slim.conv2d`和`slim.max_pool2d`的说明。

```python    
    end_points = {}
    with tf.variable_scope(scope, 'ssd_KYnet_v2', [inputs], reuse=reuse):
            # Original VGG-16 blocks.
            # net = slim.conv2d(inputs, 16, [3, 3], scope='conv1_1') #512
            net = slim.conv2d(inputs, int(32.0*net_scale), [3, 3], scope='conv2_1') #512
            end_points['block1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1') #256
            # Block 2.
            net = slim.repeat(net, 2, slim.conv2d, int(32.0*net_scale), [3, 3], scope='conv2') #256
            end_points['block2'] = net #256
            net = slim.max_pool2d(net, [2, 2], scope='pool2') #128
            # Block 3.
            net = slim.repeat(net, 3, slim.conv2d, int(48.0*net_scale), [3, 3], scope='conv3') #128
            end_points['block3'] = net #128
            net = slim.max_pool2d(net, [2, 2], scope='pool3') #64
            # Block 4.
            net = slim.repeat(net, 2, slim.conv2d, int(64.0*net_scale), [3, 3], scope='conv4') #64
            net = slim.conv2d(net, 96, [3, 3], scope='conv4_3') #64
            end_points['block4'] = net #64
```
接着上层 定义 1层128通道3x3卷积 => 1层2x2最大池化 => 1层128通道3x3卷积 => 2x2最大池化 => 1层128通道3x3卷积 => 2x2最大池化 => 1层128通道1x1卷积 => 1层2x2最大池化
```python
            # Additional SSD blocks.
            # Block 6: let's dilate the hell out of it!
            net = slim.conv2d(net, int(128.0*net_scale), [3, 3], scope='conv6') #64
            net = slim.max_pool2d(net, [2, 2], scope='pool2') #32
            end_points['block7'] = net #43

            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(128.0*net_scale), [3, 3], scope='conv3x3') #32
                net = slim.max_pool2d(net, [2, 2], scope='pool') #16
            end_points[end_point] = net #16
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(128.0*net_scale), [3, 3], scope='conv3x3') #16
                net = slim.max_pool2d(net, [2, 2], scope='pool') #8
            end_points[end_point] = net
            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, int(128.0*net_scale), [1, 1], scope='conv1x1') #8
                net = slim.max_pool2d(net, [2, 2], scope='pool') #4
            end_points[end_point] = net
```
上面代码中我们定义了SSD的主干网络，下面我们来定义分支层的结构。

```python
            # Prediction and localisations layers.
            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = ssd_multibox_layer(end_points[layer],
                                              num_classes,
                                              anchor_sizes[i],
                                              anchor_ratios[i],
                                              normalizations[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)

            return predictions, localisations, logits, end_points
```
SSD是基于不同分支上的特征图进行预测的，每个分支包含一个预测物体类别的卷积层(conv_cls)和一个负责预测物体位置的卷积层(conv_loc)，其中conv_cls预测每个Anchor的类别，而conv_loc预测检测框对于Anchors的偏差值。因此分支上的两个网络的输出维数是与该分支上的Anchor数量(num_anchors)相关的。

```python
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
```
## 5. 进行神经网络训练
### 5.1 原理介绍

完成训练集数据处理，图像预处理函数定义和神经网络函数定义后，可以通过运行`train_ssd_network.py`对我们已有的网络进行训练。训练之前，务必需要对数据集文件路径，输出模型文件路径进行正确配置。

在`train_ssd_network` 46行，确认输入`./logs_test/'`作为训练过程中的检查点文件的输出文件夹。
```python
tf.app.flags.DEFINE_string(
    'train_dir', './logs_test/',
    'Directory where checkpoints and event logs are written to.')
```

在`train_ssd_network` 140行，确认输入`test_release`作为数据集的名称。
```python
tf.app.flags.DEFINE_string(
    'dataset_name', 'test_release', 'The name of the dataset to load.')
```
在`train_ssd_network` 147行，请将`INFO_path`替换为INFO数据文件的绝对路径，保证正确的数据集路径。
```python
tf.app.flags.DEFINE_string(
    'dataset_dir', 'INFO_path', 'The directory where the dataset files are stored.')
    # change the defaut dataset_dir to local directory with INFO data.
```

在使用CPU进行训练的情况下，会得到如下结果：

```shell
2018-11-29 10:41:35.929852: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use:
AVX AVX2
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Starting Session.
INFO:tensorflow:Saving checkpoint to path ./logs_test/model.ckpt
INFO:tensorflow:Starting Queues.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Recording summary at step 0.
INFO:tensorflow:global step 10: loss = 966542.6875 (11.725 sec/step)
INFO:tensorflow:global step 20: loss = 161.3684 (12.124 sec/step)
INFO:tensorflow:global step 30: loss = 56.6930 (9.073 sec/step)
INFO:tensorflow:global step 40: loss = 60.9939 (9.895 sec/step)
INFO:tensorflow:global step 50: loss = 40.8773 (10.166 sec/step)
INFO:tensorflow:global_step/sec: 0.0900846
INFO:tensorflow:Recording summary at step 55.
INFO:tensorflow:global step 60: loss = 32.6265 (9.873 sec/step)INFO:tensorflow:global step 70: loss = 29.1204 (9.INFO:tensorflow:global step 80: loss = 21.6570 (9.INFO:tensorflow:global step 90: loss = 21.9894 (10INFO:tensorflow:global step 100: loss = 17.8053 (1INFO:tensorflow:global 
```

如果使用CPU，训练时间需要3-4个小时完成。通常情况下，当loss值下降到一定程度，不继续下降之后，可以认为loss函数收敛，训练已经完成。

训练完成后，在`./log_test/`文件夹下会有四个文件，分别为：
checkpoint：记录训练过程中中间节点保存的模型名称
model-\*.data：保存模型变量值。
model-\*.index：保存变量值和图形结构对应关系。
model-\*.meta: 保存图形结构。

这四个文件在后续实验步骤中有使用，可以被Tensorflow使用，可用来恢复神经网络模型。



### 5.2 操作流程

本教程中提供了两种格式转换方法，在IDE界面中进行操作和在终端界面中进行操作。

**注意**：需要对示例提供的代码中数据集的路径进行修改，将其指向你实际的路径。

#### 5.2.1 在IDE界面中进行操作

打开一个终端，在终端中输入`anaconda-navigator`后，便可启动Anaconda。在Anaconda中将环境切换为TensorFlow后，点击Launch启动VSCode进行操作。

具体流程可参见教程1。

在VSCode中打开`train_ssd_network.py`文件，在运行前需要确认VSCode的Python插件已被激活。


之后进行Python解析器的选择和Python环境路径的确认，具体流程可参见教程1。
用户需要对代码中第147行的数据库路径进行修改，指向用户实际的数据库路径。

![avatar](https://i.loli.net/2019/01/25/5c4ae210107eb.png)

确认无误后，可单机右键，选择在终端中运行Python文件，即可运行模型训练程序。



#### 5.2.2 在终端界面中进行操作

打开终端，输入`source activate 环境名称`，其中环境名称指的是在Anaconda中创建的TensorFlow换环境名称，可以回顾教程1的1.1.2节，在示例中环境名称为Tensorflow。

在输入命令之后，可以发现用户名前面出现了TensorFlow的环境名称，说明Tensorflow的环境已被成功开启，示意如下。
```shell
username@username-VirtualBox:~$source activate Tensorflow
(Tensorflow)username@username-VirtualBox:~$ 
```


之后，在Tutorial2_codes下目录中运行如下代码，需要注意的是，我们也需要指明脚本中的数据库路径。

```shell
sh script/2_run_training.sh
```

即可执行脚本，进行模型的训练。可以通过`cat script/2_run_training.sh`来浏览脚本中的代码，代码内容如下。代码中具体数值所表示的含义将会在后续的课程中介绍。
```shell
DATASET_DIR=./ #指定数据库路径
TRAIN_DIR=./logs_test/ #指定保留checkpoint文件的路径
DATASET_NAME=test_release #明确TF格式数据库的名称


if [ ! -d ${TRAIN_DIR} ];
then mkdir ${TRAIN_DIR}
fi

cp ${DATASET_DIR}/${DATASET_NAME}.json ${TRAIN_DIR}/image.json
cp ${DATASET_DIR}/${DATASET_NAME}.INFO ${TRAIN_DIR}/image.INFO

python -u train_ssd_network.py \
      --gpus=1 --num_clones=1 \
      --train_dir=${TRAIN_DIR} \
      --dataset_dir=${DATASET_DIR} \
      --dataset_name=test_release \
      --dataset_split_name=train \
      --model_name=ssd_KYnet_v2_5b \
      --save_summaries_secs=600 \
      --save_interval_secs=1200 \
      --weight_decay=0.00001 \
      --optimizer=adam \
      --learning_rate=0.005 \
      --learning_rate_decay_factor=0.95 \
      --batch_size=20 --debug_type=True \
      --num_epochs_per_decay=100.0

```



## 6. 进行图片测试验证结果

### 6.1 原理介绍

本步骤的目标是使用训练好的模型文件对图片进行推理从而得到人脸的位置信息。本步骤中需要使用一个全新的python图像处理库`OpenCV`, 关于`OpenCV`的介绍请参考[本链接](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_intro/py_intro.html)。`OpenCV`是专用于计算机图像处理领域的库。

**测试过程主要包括以下步骤**
> 1. 输入图像 --> 3D-array
> 2. 图像前处理 --> 网络输入
> 3. 网络前向inference --> 网络输出feature map
> 4. Box decoding --> 候选框位置
> 5. 根据Score排序 --> Score从高到底的候选框列表
> 6. 非极大值抑制（NMS）--> 最终检测结果


在linux环境下可以使用一下命令安装OpenCV
```shell
pip install opencv-python
```

进入`run_single_image.py`, 添加一下功能代码，实现

首先要先确定模型模型路径：
如果使用上一步中训练的模型检测单张图片的人脸结果，请使用以下路径。本测试的结果取决于进行网络训练的时间。如果训练时间很短，人脸检测的精度会比较差。此时需要使用提前训练好的模型文件。
```python
f.app.flags.DEFINE_string('_model_path', './log_test/','model_path')
```

如果需要使用提前训练好的神经网络检查结果，请使用`/pretrain/ssd_5b/`下的模型文件和`checkpoint`文件。
```python
f.app.flags.DEFINE_string('_model_path', './pretrain/ssd_5b/','model_path')
```

定义函数查找定义的模型文件中的checkpoint文件，Tensorflow可以根据神经网络模型路径，恢复训练好的模型，在`run_single_image.py`中第40行，文件读取请参考[python input&output操作](https://docs.python.org/3/tutorial/inputoutput.html), 路径查找请参考[OS 操作](https://docs.python.org/2/library/os.path.html)。

输入以下代码，查询checkpoint文件：

```python
def parse_model_name(model_dir):
    CHECK_POINT_FILE = "checkpoint"
        if os.path.isdir(model_dir):
            with open(os.path.join(model_dir, CHECK_POINT_FILE), 'r') as f:
                first_line = f.readline().strip()
            if first_line:
                return os.path.join(model_dir, first_line.split(' ')[1].strip('"'))
        raise RuntimeError(
            "Failed to find the _model_path directory %s" % FLAGS._model_path)
```

使用`cv2.read`函数将需要进行检测的图片导入程序中，然后将图片输入神经网络，随后得到输出结果，同时计算图片处理时间`time cost`，可在第112行读取图片，进行图片处理。

```python
    img = cv2.imread(FLAGS._img_path)
    img_tobe_processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("image shape", img.shape)
    time_start = time.clock()
    rclasses, rscores, rbboxes = process_image(img_tobe_processed, select_threshold=0.25, net_shape=net_shape)
    time_end = time.clock()
    print('totally time cost', time_end - time_start)
```

### 6.2 操作流程

#### 6.2.1 在IDE界面中进行操作

打开一个终端，在终端中输入`anaconda-navigator`后，便可启动Anaconda。在Anaconda中将环境切换为TensorFlow后，点击Launch启动VSCode进行操作。

具体流程可参见教程1。

在VSCode中打开`run_single_image.py`文件，在运行前需要确认VSCode的Python插件已被激活。


之后进行Python解析器的选择和Python环境路径的确认，具体流程可参见教程1。确认无误后，可单机右键，选择在终端中运行Python文件，即可运行模型训练程序。

#### 6.2.2 在终端界面中进行操作
在Tutorial2_codes下目录中运行如下代码：

```shell
sh script/3_run_test.sh
```

即可执行脚本，进行模型的训练。可以通过`cat script/2_run_training.sh`来浏览脚本中的代码。
查看脚本，可以发现该脚本调用了`run_single_image.py`文件，并向其传入了三个参数，`_model_key`为模型名称，在路径`log_test`中存放着该模型的chekpoint文件，图像输入的路径为`_img_path`，用户可以通过修改输入图像路径，选择其他图像进行测试。如下图所示。
![avatar](https://i.loli.net/2019/01/25/5c4ad87b36562.png)


运行程序之后在终端中显示的结果应为：

![avatar](https://i.loli.net/2019/01/25/5c4a85f410258.png)


图像检测的结果应如下图所示：

![avatar](https://i.loli.net/2019/01/25/5c4ad50477014.png)


## 7. cv2模块加载失败的解决方案
在执行上述格式转换、模型训练、模型验证的示例时，可能会出现`import error dll load failed cv2`的报错，这是因为OpenCV的版本与Python不匹配导致的，可以通过以下代码更新OpenCV解决。
```
pip install opencv-contrib-python
```

## 8. 练习

独立加入图片前处理代码，实现步骤6中的相同结果。请打开`run_single_image_exercise.py`，本文件已经将对图片预处理的过程移除。需要单独使用`OpenCV`进行图片预处理。
请自己在第105行添加图片预处理过程。
 >  具体步骤如下：
    1. 读取图像 --> 使用cv2.imread()
    2. 调整通道顺序 --> BGR转RGB，使用cv2.cvtColor()
    3. 显示进行BGR -> RGB转换后图像的区别。可以使用`cv2.imshow`显示图片。
    4. 图像缩放到256x256 --> 使用cv2.resize()。展示resize后图片和原始图片的区别。可以使用`cv2.imshow`。
    5. 图像消减mean --> 参考numpy示例，R通道-123，G通道-117， B通道-104 


## 9. 附录：参数说明

`通用参数`

+ `dataset_name`: 训练中使用的数据库名称，应与数据转换时设定的名称保持一致。
+ `dataset_dir`: 数据存储的路径。
+ `model_name`: 选择使用的网络模型结构，定义了SSD的前向传播基础网络。可从`datasets/dataset_factory.py`查看可选网络列表。用户在添加自己的网路时，也需要在`dataset_factory.py`中声明自定义网络名称。
+ `checkpoint_path`: 如果用户希望从已有网络模型中fine-tune，需要指定已有模型路径。
+ `train_dir`: Tensorflow Summary和模型文件输出路径。
+ `log_every_n_steps`: 每隔多少步在命令行中打印和显示日志。
+ `save_interval_secs`: 每隔多少秒存储一次模型文件，时间越短，存储的中间模型数量越多，也会占用较多的硬盘资源。

``GPU训练相关参数``

+ `gpus`: 训练接口支持多GPU并行计算。该选项仅适用于使用GPU的用户，输入空闲的GPU id，用逗号隔开。如1，2，3，4。

+ `num_clones`: 设置为训练中使用的GPU数量。


``优化相关参数``

+ `max_number_of_steps`: 最大训练迭代数。我们将一次forward+backward propagation的计算称为一个step。该参数定义了将网络迭代更新多少步。通常来说，我们需要设置一个较大的值来保证训练结束前，网络已经收敛了，即loss不再继续下降。
+ `batch_size`: 训练过程中，通常将训练数据拆分成多个data batch，而每个batch中样本的数量称为batch_size, 一般来说在显存允许的情况下，batch_size越大越好。
+ `optimizer`: 优化器选择，可选项包括`adadelta`,`adagrad`,`adam`,`ftrl`,`momentum`,`sgd`和`rmsprop`。不同优化器的网络参数更新方式和学习率调整策略有所不同，本实验中建议使用`adam`。
+ `weight_decay`: 正则项系数，通常<1。正则项通过将网络参数的l2_norm加入最终loss中来抑制过拟合现象，通过调整该系数保证正则项起到作用，但是又不会过大地影响问题本身的loss。通常来说，通过该系数调整，正则项loss为总loss的1/5~1/10较为常用。


``学习率相关参数``

+ `learning_rate`: 起始学习率。起始学习率过大，可能导致NaN数值错误。起始学习率过小，可能导致收敛过慢。

+ `learning_rate_decay_factor`: 每次学习率调整时的调整幅度，通常<1以保证learning_rate逐渐减小。

+ `num_epochs_per_decay`: 每隔多少步调整学习率。

这三个参数定义了学习率的更新规则，即每隔`num_epochs_per_decay`步，`learning_rate`变为`learning_rate_decay_factor`x当前`learning_rate`。

Tutorial 2 到此结束。