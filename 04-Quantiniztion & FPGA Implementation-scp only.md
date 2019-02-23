
# 神经网络压缩与算法部署流程


本实验教程分为三部分，第一部分为神经网络压缩的基础概念介绍，第二部分为编译器Rainbuilder的工作流程和原理，第三部分为经过编译的算法模型在FPGA上的部署流程。

<!-- TOC -->

- [神经网络压缩与算法部署流程](#神经网络压缩与算法部署流程)
    - [1. 神经网络压缩基础](#1-神经网络压缩基础)
        - [1.1 Floating-Point vs. Fixed-Point](#11-floating-point-vs-fixed-point)
        - [1.2 低位宽压缩](#12-低位宽压缩)
    - [2. RainBuilder 8-bit 流程](#2-rainbuilder-8-bit-流程)
        - [2.1 Plumber 8-bit流程](#21-plumber-8-bit流程)
            - [2.1.1 量化模型](#211-量化模型)
            - [2.1.2 生成SG](#212-生成sg)
            - [2.1.3 优化SG](#213-优化sg)
            - [2.1.4 为硬件优化SG](#214-为硬件优化sg)
            - [2.1.5 导出数据](#215-导出数据)
        - [2.2 Raintime 8-bit流程](#22-raintime-8-bit流程)
            - [2.2.1 示例编译](#221-示例编译)
    - [3. 板卡验证](#3-板卡验证)
        - [3.1 配置虚拟机环境并登录](#31-配置虚拟机环境并登录)
        - [3.2 板卡环境搭建](#32-板卡环境搭建)
        - [3.3 文件导入](#33-文件导入)
        - [3.4 在板卡上执行模型](#34-在板卡上执行模型)
    - [练习1：不使用实际场景的图片结果](#练习1不使用实际场景的图片结果)

<!-- /TOC -->


## 1. 神经网络压缩基础
### 1.1 Floating-Point vs. Fixed-Point
在计算机系统中，表示一个非整数通常由floating-point和fixed point两种表达形式。Floating-point represenation中表示整数部分和小数部分的bit数是动态的，也就是说对于不同的数值，这两部分的数据位数是变化的。而fixed-point representation整数和小数部门的bit位数是静态固定的。设计一个基于floating-point的高性能架构的难度较大，对于存储空间和计算资源的要求相关，因此当前绝大多数低功耗、高性能的计算架构采用fixed-point representation。关于floating-point和fixed-point比较请参考下图，
 
![Floating-Point vs. Fixed-Point](https://i.loli.net/2019/02/12/5c62306284275.png)

由于fixed-point representation中整数、小数位宽是静态指定的，在设计计算架构师需要提前确定两者的分配。通常来说，开发者需要提前确定所有数据的范围，如最大值和最小值，来确定整数部分所需的最小位宽，进而确定小数位宽。架构在使用fixed-point的同时会引入误差，通常数据表示位宽越小，误差越大。

### 1.2 低位宽压缩
CNN的计算量通常比较大，不适合资源有限的嵌入式系统。低位宽（low-bit）压缩将原来基于float32的算术操作转化为low-bit fixed-point(e.g. INT8)计算，从而大大减小计算量和存储空间，进而提升CNN的运行速度和效率。

量化(quantization)作为一种常用的low-bit压缩的方法，被广泛应用于很多现有计算框架中。量化的目的是将连续信号近似为有限多个离散值的过程，也就是说将输入的数值映射到一个固定范围的数值中去的过程，而其中从量化值恢复得来的数值和原始数值之间的区别称为量化误差。

我们假设待量化的输入为$r\in[a,b]$, 量化输出$q$使用$n-bit$表示, 量化阶数$M=2^n$, 则量化数学表达式如下，

$q = {\lfloor} \frac{r}{s} + Z{\rceil}$ , 其中$s=\frac{b-a}{M-1}$，$Z=-\frac{a}{s}$，$\lfloor\cdot\rceil$表示四舍五入操作。

例如，我们要将$1,2,3,4,5,6,8$使用$2-bit$进行量化压缩，那么最终结果如下

![quant](https://i.loli.net/2019/02/12/5c622ee442f1d.png)。

## 2. RainBuilder 8-bit 流程
Rainbuilder包含了Plumber和Raintime两个模块，Plumber是用于生成基于FPGA的高性能CNN推理系统的工具链，以高级CNN描述和TensorFlow的checkpoint文件为输入，输出FPGA硬件可识别的模型描述文件。Raintime则是鲲云FPGA板卡上的计算图运行时（raintime）和卷积神经网络计算库，它接收plumber输出的SG描述和权值文件，根据SG描述部署计算过程并执行计算。
本章节将详细介绍Plumber和Raintime的8bit量化流程。

**注**：本实验提供了与训练好了的人脸识别算法`face_model_5b`，位于docker中`/workspace/8bit/`路径下。

为了方便描述，我们将Plumber的输出结果路径(即下文中的`${QUANTIZE_PATH}`)设置为`/workspace/8bit/ssd_face_model256`。

为了简化用户的使用流程，我们预先编写了一个操作脚本`run_face.sh`，放置在在`/workspace/8bit/`路径下，该脚本包含了上述所有plumber对模型进行8bit量化的命令，用户可通过执行此脚本来完成2.1.1至2.1.5的所有步骤。

接下来我们将详细介绍部署算法的详细流程。

### 2.1 Plumber 8-bit流程
8bit Plumber与之前的16bit版本相比，流程上有一些改变。
8bit流程包括 `量化->生成SG->优化SG->硬件优化->导出数据` 五个步骤。

#### 2.1.1 量化模型
该步骤为8bit plumber的主要改动。

Plumber中使用以下命令完成量化操作
```shell
>>> plumber_cli quant --help

Usage: plumber_cli quant [OPTIONS] MODEL_DIR

  Quantizion a TensorFlow model from checkpoint files.

Options:
  --image-dir PATH              图像数据集路径
  -d, --output-dir PATH         结果输出路径
  -o, --output-node-names TEXT  输出节点名，使用逗号分隔。如不指定该项，plumber会
                                列出自动识别出的输出节点名称，用户可从中按序号进行
                                选择
  --preprocess-call TEXT        数据前处理函数定义
  --help                        显示帮助信息
```

使用示例如下
```shell
plumber_cli quant ${CKPT_PATH} --image-dir ${FDDB_PATH} -d ${QUANTIZE_PATH} --preprocess-call ${PREPROCESS_SCRIPT}
```
其中`${CKPT_PATH}`为模型文件所在路径。

在8bit版本中，用户需要指定输入图片数据集路径和数据前处理函数定义完成量化操作。对于CNN网络的量化主要包括对于模型参数，如卷积weight和bias，和中间层输出结果的量化。为了实现量化，量化模块需要提前确定每个待量化参数的值域范围，即最大值和最小值。Plumber中是通过输入网络一定量的图像数据来计算并统计每个参数的值域，因此需要提供文件夹的路径。我们提供的示例中使用[FDDB](http://vis-www.cs.umass.edu/fddb/)数据库，``/workspace/8bit/``路径下的`fddb`文件夹即为本例所需的图像数据库。用户需要定义输入数据的前处理函数，因为前处理函数的不同会导致输入网络的数据范围改变，进而改变网络中间层的输出值域。

在本步骤中的命令在`/workspace/8bit`路径下运行，为：
```
plumber_cli quant ./face_model_5b/ --image-dir ./fddb/ -d ./ssd_model_face256 --preprocess-call preprocess_ssd.py
```
输出结果如下图所示：

![freeze_model](https://i.loli.net/2019/02/14/5c64d4cdc47a6.png)

选择需要输出的节点后，程序会自动运行，最后在`/ssd_model_face256`中生成`8-bit`、`model.pb`、`quant_ckpt`三个文件。

注意：quant步骤中为了统计每层的数据值域，需要使用网络计算所有数据库中的图像，我们建议在该步骤使用GPU。

前处理函数的定义可参考如下例子，用户可根据自己的需求编写自己所需的前处理函数。
```python
def generate_img(fddb_root):
  # 遍历数据库中的所有数据
  for i in range(1, 11):
    input_path = os.path.join(fddb_root, 'FDDB-folds/FDDB-fold-%02d.txt' % i)
    # 读取FDDB某一fold中的所有数据
    with open(input_path, 'r') as f:
      for line in f.readlines():
        img_name = line.strip()
        img_path = os.path.join(fddb_root, img_name + '.jpg')
        img = cv2.imread(os.path.join(fddb_root, img_name+'.jpg'))
        """
        数据前处理：
        1. 图像resize到256x256
        2. 数据减mean操作
        """
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = np.array([123.0, 117.0, 104.0])
        img = img - mean
        # 将图像从三维拓展到四维（跟模型裁剪时的placeholder维数保持一致）
        new_shape = [1] + list(img.shape)
        img = img.reshape(new_shape)
        # 返回generator，通过调用generator的next函数完成数据的迭代
        yield img
```

函数需命名为`generate_img`，并包含数据集路径作为参数进行定义。
函数中需要完成两项内容：1）对于数据库结构的解析和遍历 2）每个输入数据的前处理方法。



#### 2.1.2 生成SG
该步骤将Tensorflow的模型文件`*.pd`转化为SG的表达式。
``` shell
plumber_cli sg -d ./ssd_model_face256
```
参数`-d`用于指定输出路径，同时要求输出路径中还有上一步生成`model.pb`文件，因此此步骤中的输出路径需与2.1.1量化步骤中的输出路径保持一致。

本步骤的输入命令依然在`/workspace/8bit`路径下运行，为
```
plumber_cli sg -d ./ssd_model_face256 --input-value-range -123,151
```

运行命令后得到下面的结果，使用SG表达的神经网络结构图：
```shell
node {
  name: "img_input"
  op: "Input"
  device: CPU
  type: T_FLOAT
  input_op_param {
    shape {
      dim: 1
      dim: 3
      dim: 512
      dim: 512
    }
  }
}
...
node {
  name: "ssd_KYnet_v2/conv1_1/Conv2D_weights"
  op: "Const"
  device: CPU
  type: T_FLOAT
  data_map {
    key: "data"
    value {
      shape {
        i: 16
        i: 3
        i: 3
        i: 3
      }
    }
  }
}
node {
  name: "ssd_KYnet_v2/conv1_1/Conv2D_bias"
  op: "Const"
  device: CPU
  type: T_FLOAT
  data_map {
    key: "data"
    value {
      shape {
        i: 16
      }
    }
  }
}
...
node {
  name: "ssd_KYnet_v2/conv1_1/Conv2D"
  input: "img_input"
  input: "ssd_KYnet_v2/conv1_1/Conv2D_weights"
  input: "ssd_KYnet_v2/conv1_1/Conv2D_bias"
  op: "Conv2D"
  device: CPU
  type: T_FLOAT
  conv2d_op_param {
    depth: 16
    kernel_size: 3
    pad: 1
    stride: 1
    activation_fn: "Relu"
    use_maxpool_2x2: false
    use_batch_norm: false
    use_bias: true
    use_relu: true
  }
}
...
```
其中op为Input类型的节点代表网络的输入，op为Const的节点通常代表网络中参数，如weight、bias等。其余为常规计算节点，通过input字段表示该节点的输入节点，从而表示整个网络中不同层的连接关系。

#### 2.1.3 优化SG
优化SG主要对模型进行数据精度和稀疏性分析。只需要指定plumber `sg` 命令编译后的文件夹路径，该文件夹路径包含了`sg`命令生成的所有文件。
```shell
plumber_cli sg_opt -d ${OUTPUT_PATH} --quant-model ${QUANTIZE_PATH}/8-bit.pb --input-min-max ${INPUT_RANGE}
```
其中`--quant-model`为量化步骤中生成的pb文件路径，`--input-min-max`为输入数据的理论最大值和最小值。

我们以步骤1中的前处理方法为例，输入数据$\in [0,255]$，前处理包括了resize操作和减mean操作。其中resize操作不会改变数据值域，因此resize过后的范围仍然是$[0, 255]$。减mean操作中，三通道的mean值分别为$123、117、104$，因此输出最小值为$0-123=-123$，最大值为$255-104=151$，对应的调用参数为`--input-min-max -123,151`。

本步骤中的指令运行路径与上一步的路径保持一致，为：
```
plumber_cli sg_opt -d ./ssd_model_face256 --quant-model ./ssd_model_face256/8-bit.pb --input-min-max -123,151
```


#### 2.1.4 为硬件优化SG
`-d ${QUANTIZE_PATH}`为之前几个命令步骤中间临时文件的存放路径，`${BOARD_FILE}`是板卡参数定义文件，针对不同的板卡，硬件优化执行不同的优化操作，并分配不同的硬件执行设备。在步骤中会生成`model_hdl_sg.pbtxt`文件，它将作为`Raintime`的模型输入。
其中`${BOARD_FILE}`文件为docker中`/workspace/8bit/`路径下的`rainman_board_v3_8bit.pbtxt`文件

```shell
plumber_cli hdl_opt -d ${OUTPUT_PATH} ${BOARD_FILE}
```

在此步骤中执行3个优化过程。

|步骤名称|功能|
|:-:|:-:|
|融合|将多个算子融合成一个硬件计算模块|
|设备|使用执行设备标记SG节点|
|设计|设计空间探索|

具体指令为：
```
plumber_cli hdl_opt -d ./ssd_model_face256 rainman_board_v3_8bit.pbtxt
```


#### 2.1.5 导出数据
命令使用示例如下：
```shell
plumber_cli export -d ${OUTPUT_PATH} -t float_little --use-hdl True -pf 8
```
导出数据中`weights`为8bit fixed数据，`bias`为32bit fixed数据，其他为floating-point数据。
数据导出的最终结果存储在`${OUTPUT_PATH}/data_hdl/float_little/`文件夹中。

本步骤的具体指令为：
```
plumber_cli export -d ./ssd_model_face256 -t float_little --use-hdl True -pf 8
```

### 2.2 Raintime 8-bit流程

#### 2.2.1 示例编译
用户所有源码的编译均在docker中完成。docker中已经配好了编译环境，保证编译得到的可执行文件可以在板卡端直接运行。

我们推荐使用CMake辅助编译过程。CMake需要编写对应的CMakeList.txt文件定义编译环境配置。我们需要在CMakeList中配置交叉编译环境和板卡运行时环境，代码如下所示。
```shell
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CROSSCOMPILING ON)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
# System Root Path
set(CMAKE_SYSROOT /workspace/toolchain/sysroot/)
# Cross Compiler path 
set(CMAKE_C_COMPILER /workspace/toolchain/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /workspace/toolchain/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++)

# raintime include path
include_directories(
  /workspace/install/include/
  )

# raintime lib path
link_directories(
  /workspace/install/lib/)

```
我们已经将预先写好的CMakeList在`/workspace/8bit/demo_face_5b/`路径下，用户只需在此路径下依次执行
```
cd build 
cmake ..
make -j
```
即可得到用于调配FPGA板卡资源的可执行文件。

## 3. 板卡验证
完成了2.1 Plumber 8-bit流程后，我们得到了FPGA硬件可识别的模型描述文件（.pbtxt）和模型参数文件。完成2.2 Raintime 8-bit流程之后，我们则得到了可以调用FPGA板卡资源的可执行文件。接下来，在此章节中，我们将详细介绍如何在板卡上部署算法，并使用板卡运行算法模型。


### 3.1 配置虚拟机环境并登录
按照教程《软件与软件环境安装手册》第三章FPGA板卡配置与登录的2.1 网络设置与板卡登陆小节进行虚拟机网络配置，将虚拟机与板卡配置在同一个网段中。

之后，打开虚拟机的网络设置，将网卡的连接方式从网络地址转换（NAT）更改为**桥接网卡**，并选择自己计算机上正确的界面名称。如此，便能通过网口与板卡进行连接。
![choose-nw-card](https://i.loli.net/2019/02/13/5c63d548f0241.jpg)
在配置好主机系统的网络参数后，在Linux中打开命令行，使用以下命令登录板卡：

```shell
ssh root@192.168.123.8
```
输入密码：
```shell
letmein
```

### 3.2 板卡环境搭建

在通过`ssh`进入板卡后，因为在实验三中我们将板卡环境切换为了16bit，因此在此实验中我们需要把板卡切换回8bit。具体操作如下：
```shell
cd /workspace
./switch_hw_rbf.sh 8      #执行指令后板卡会自动重启，需要重新ssh登录进板卡
./load_raintime8.sh       #加载8bit环境
```

Rainman板卡专用于高性能深度学习的推演计算，因此只包含运行环境和有限的板上存储空间。

开发过程中，用户所有的数据和临时文件都直接存放在本地PC上，通过`scp`命令，实现板卡端和PC端的数据传输和共享。




### 3.3 文件导入
回到docker中，本教程第二章生成的描述文件`model_hdl_sg.pbtxt`和可执行程序为`demo_runner_face5b`分别存放在docker中的`/workspace/8bit/ssd_face_model256/`和`workspace/8bit/demo_face_5b/build/`目录下。

同时，由于示例中使用的是SSD算法，需要对图像进行后处理操作，所以后处理文件`/workspace/8bit/param`也需要拷贝至板卡。

综上，我们需要将`model_hdl_sg.pbtxt`文件夹、`demo_runner_face5b`程序和`param`文件夹总共三分数据拷贝至板卡中。

通过在虚拟机中执行以下命令，即可将所需的文件拷贝至虚拟机的`/home/tutorial/board_share`目录中。
```
sudo docker cp plumber_env_cpu:/workspace/8bit/ssd_face_model256/model_hdl_sg.pbtxt /home/tutorial/board_share 
sudo docker cp plumber_env_cpu:/workspace/8bit/demo_face_5b/build/demo_runner_face5b /home/tutorial/board_share 
sudo docker cp plumber_env_cpu:/workspace/8bit/param /home/tutorial/board_share 
```
之后，登录至板卡，在板卡中输入以下命令将上述三个文件拷贝至板卡中。
```
scp -r tutorial@{VM_IP}:/home/tutorial/board_share/ssd_model_face256 /workspace/share
scp -r tutorial@{VM_IP}:/home/tutorial/board_share/param /workspace/share
scp tutorial@{VM_IP}:/home/tutorial/board_share/demo_runner_face5b /workspace/share

```

### 3.4 在板卡上执行模型
将测试图片拷贝至虚拟机的`/board_share`文件夹中，同样使用scp命令拷贝，并在板卡上执行以下命令
```
./workspace/share/demo_runner_face5b \
--pbtxt /workspace/share/model_hdl_sg.pbtxt \             #模型描述文件
--coeff_path /workspace/share/data_hdl/float_little \     #模型参数文件
--param_path /workspace/share/param/ \                    #图像后处理文件
--input_path ./1.jpg \                                    #测试图像
--output_path ./out.jpg \                                 #输出结果
--net_class 2                                             #类别数=识别类别总数+1
--sim_only=false                                          #开启真实模式
```
输出结果如下图所示，并保存在`/workspace/share/`路径下：

![result](https://i.loli.net/2019/02/13/5c63e7aa5e955.jpg)

用户可以通过在板卡中输入以下指令将图像结果拷贝至虚拟机本地进行查看。
```
scp -r /workspace/share/out.jpg toturial@{VM_IP}:/home/tutorial/
```

## 练习1：不使用实际场景的图片结果

假设在进行8bit量化的过程中，使用随机图片，比如黑，绿，红，白：
![pic.PNG](https://i.loli.net/2019/02/22/5c6f8111ca5b8.png)

使用板卡进行图片检测会发生什么？