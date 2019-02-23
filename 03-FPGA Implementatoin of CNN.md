# 基于SSD框架的目标检测算法的FPGA部署

本课程我们介绍用于如何使用编译器将人工智能CNN算法固化到FPGA上，本节实验课的内容分为以下几个部分：
1. 学习和理解编译过程，可以帮助理解计算机和FPGA的编译。
2. 学习如何使用Plumber编译器将人脸检测算法部署到FPGA。
3. 练习1：将行人检测网络部署在FPGA并验证结果。
4. 目前检测算法后处理介绍。
5. 练习2：实现后处理并观测结果。


  - [1. 在FPGA上部署人脸检测算法](#1-在fpga上部署人脸检测算法)
      - [1.1 编译过程原理](#11-编译过程原理)
      - [1.2 名词解释](#12-名词解释)
  - [2. 学习如何使用Plumber编译器将人脸检测算法部署到FPGA](#2-学习如何使用plumber编译器将人脸检测算法部署到fpga)
      - [2.1 固化模型](#21-固化模型)
          - [其他命令选项](#其他命令选项)
      - [2.2 生成SG](#22-生成sg)
      - [2.3 优化SG](#23-优化sg)
      - [2.4 为硬件优化SG](#24-为硬件优化sg)
      - [2.5 导出数据](#25-导出数据)
      - [2.6 将FPGA需要的数据整理到相同文件夹](#26-将fpga需要的数据整理到相同文件夹)
      - [2.7 配置板卡并将需要使用的相关文件上传至板卡](#27-配置板卡并将需要使用的相关文件上传至板卡)
          - [2.7.1 硬件设置与板卡登录](#271-硬件设置与板卡登录)
          - [2.7.2 Docker文件向板卡的传输](#272-docker文件向板卡的传输)
      - [2.8 登录板卡并使用FPGA运行人脸检测程序](#28-登录板卡并使用fpga运行人脸检测程序)
          - [2.8.1 板卡登录](#281-板卡登录)
          - [2.8.2 运行人脸检测程序：](#282-运行人脸检测程序)
  - [练习1： 请使用行人检测模型对图片实现行人检测](#练习1-请使用行人检测模型对图片实现行人检测)
  - [目前检测算法后处理介绍](#目前检测算法后处理介绍)
  - [练习2：实现后处理并观测结果。](#练习2实现后处理并观测结果)



**实验内容**：

## 1. 在FPGA上部署人脸检测算法

在上个实验中，我们已经介绍如何使用Tensorflow搭建基本的神经网络，并且进行基于SSD算法的训练过程。本次实验将会介绍如何将训练成功的算法部署在FPGA中，并验证算法效果。

### 1.1 编译过程原理
**Plumber**是用于生成基于FPGA的高性能CNN推理系统的工具链。Plumber以高级CNN描述和TensorFlow的训练数据为输入，输出一个经过优化的可运行的*软件程序*以及*硬件设计*。

在第二步中，我们已经部署好了集成Plumber的环境。Plumber的一个重要功能是将Tensroflow结构图转化为Streaming Graph(SG）数据图的表达形式，随后将SG传递到FPGA和CPU中运行。 
Plumber是一种多级编译方式，可以从高级语言直达FPGA底层逻辑。

Plumber多级编译方式如下图所示：
![5c0a2c24b3ead.png](https://i.loli.net/2018/12/11/5c0f322747f7c.png)

本次实验，我们将使用在Tutorial 2中生成的checkpoint模型文件，使用Plumber编译器，一步步生成SG文件，然后将生成的文件下载进入FPGA板卡进行运行。

### 1.2 名词解释

* SGNode: SGNode是Plumber对模型内部的**一组计算**方式的高级描述，如卷积计算中有偏移加和激活函数计算，SGNode把它们包含在一起表示。
* SG: Streaming Graph, 数据流图，Plumber内部的数据结构，它由一系列SGNode组成。
* SG IR: 对SG进行序列化，生成的文件称之为SG IR(.pb, .pbtxt)，它们使用跨平台高性能的`protobuf`协议的文件格式。
* 固化模型: 使用TensorFlow训练的模型，一般会被保存为检查点文件，其中模型的参数和图结构都分散在不同的文件中，而Plumber期望得到一个把模型和参数都固定到同一个文件的模型，所以有了固化模型这个命令。

## 2. 学习如何使用Plumber编译器将人脸检测算法部署到FPGA

- [固化模型](#固化模型)
- [生成SG](#生成SG)
- [优化SG](#优化sg)
- [为硬件优化SG](#为硬件优化sg)
- [导出数据](#导出数据)

正常的模型编译需要经过四个步骤，`生成SG -> 优化SG -> 为硬件优化SG -> 导出数据`，这四个步骤是互相依赖的关系，执行的时候切记按照顺序执行。

通过以下命令启动并进入docker，密码为`letmein`:
```
sudo docker start plumber_env_cpu_16bit
sudo docker exec -ti plumber_env_cpu_16bit bash
```

成功进入docker之后，**进入`/app`路径**，此路径中有本次实验中所需的数据资料。

本实验提供了已经训练好了的人脸检测算法，并已经提前放置在docker的`/app/inference_model`文件中。本次实验将以此模型为基础进行。

本实验中提供了能够直接运行2.1至2.6全流程的脚本`0_run_complete_flow.sh`，接下来本文将介绍算法部署的详细执行步骤。

### 2.1 固化模型

由于原始的模型带有用于求解模型权值和偏移量的结构或节点，在硬件上部署的算法不需要重新求解权值和偏移量，因此这些结构或节点对于硬件来说是没有必要的。因此在将算法部署到硬件上之前，需要对模型进行类似于“剪枝”的操作，我们称为“固化”模型。

我们对TensorFlow训练过程中得到的模型文件进行固化，从而得到`*.pb`文件，固化模型的具体命令如下：
```shell
plumber_cli freeze ./inference_model/ -d ./tmp
```

第一个参数`./inference_model`在TensorFlow训练过程中被保存的检查点文件，它是一个必须要提供的参数，文件夹内应只包含以下文件：
1.	`*.meta`：有关模型的元信息
2.	`*data*`：训练数据
3.	`*.index`：索引文件

如果你的文件夹中有`checkpoint`文件，而里面的`model_checkpoint_path`的路径还是训练时的绝对路径，会导致固化模型失败。

本步骤中的`inference_model`文件夹在docker中的`./app`路径下，用户可以进入文件夹中进行查看。

第二个参数`-d ./tmp`是指定冻结后模型的保存路径，这里指定为一个文件夹，并且可以是不存在的文件夹，Plumber会自动创建。文件夹中会生成`model.pb`文件，这种文件是`protobuf`协议的文件格式。


#### 其他命令选项

`--output-file-name`

生成固化模型的文件路径。如果只想生成固化后的模型文件，可以使用此选项，如果用于编译模型，请选择`-d`指定一个文件路径的选项。

`--output-node-names`

指定输出节点名称的选项，多个输出节点使用英文逗号分隔，作为可选参数，如果不指定， 请输入`all`,Plumber会自动分析模型寻找可能的输出节点，并供你选择，如：

```
You haven't specified any output node, please select one or many from the following list:
[  0] ssd_KYnet_v2/block9_box/Reshape
[  1] ssd_KYnet_v2/softmax_1/Reshape_1
[  2] ssd_KYnet_v2/softmax_3/Reshape_1
[  3] ssd_KYnet_v2/softmax/Reshape_1
[  4] ssd_KYnet_v2/block4_box/Reshape
[  5] ssd_KYnet_v2/softmax_2/Reshape_1
[  6] ssd_KYnet_v2/block10_box/Reshape
[  7] ssd_KYnet_v2/block7_box/Reshape
[  8] ssd_KYnet_v2/softmax_4/Reshape_1
[  9] ssd_KYnet_v2/block8_box/Reshape
Please enter the indices of output nodes, separated by ','. If you want to select all, please enter 'all': all
Output node names: ['ssd_KYnet_v2/block9_box/Reshape', 'ssd_KYnet_v2/softmax_1/Reshape_1', 'ssd_KYnet_v2/softmax_3/Reshape_1', 'ssd_KYnet_v2/softmax/Reshape_1', 'ssd_KYnet_v2/block4_box/Reshape', 'ssd_KYnet_v2/softmax_2/Reshape_1', 'ssd_KYnet_v2/block10_box/Reshape', 'ssd_KYnet_v2/block7_box/Reshape', 'ssd_KYnet_v2/softmax_4/Reshape_1', 'ssd_KYnet_v2/block8_box/Reshape']
2019-02-15 08:11:54 INFO Initialising with model directory ...
Successfully writen the frozen graph to ./tmp/model.pb

```
`--use-scope-match`

布尔值，Plumber自动寻找可能的输出节点模式，默认根据节点的`Scope`来寻找，这样可以过滤许多不必要的训练图，从而让缩小正确的数据节点列表。如果你的模型中没有使用`Scope`，设置为`False`效果会更好。可以使用`-m True`或`--use-scope-match=True`。

注意: **如果忘记了它的参数选项，可以使用`plumber_cli freeze --help`来查看命令选项**


我们为此步骤提供了自动化脚本，为`1_plumber_freeze.sh`，用户可以在docker中的`/app`路径下运行此脚本。也可以通过`vi 1_plumber_freeze.sh`或`cat 1_plumber_freeze.sh`命令来观察脚本中的具体指令。

### 2.2 生成SG
该步骤将Tensorflow的模型文件`*.pd`转化为SG的表达式，具体的命令如下。
```
plumber_cli sg -d ./tmp -s 1,256,256,3
```
第一个参数`-d ./tmp`指定被固化后的模型文件路径，并且固化模型的文件名称符合`model.pb`格式。

第二参数`-s 1,256,256,3`，指定模型的输入形状（input shape）对应于TensorFlow的数据格式，根据模型输入，其输入应为`NHWC`或`NCHW`。

此步骤的脚本为`2_plumber_genSG`。

运行后得到下面的结果，使用SG表达的神经网络结构图：
```shell
Final SG: 
name: ""
node {
  name: "img_input"
  op: "Input"
  device: CPU
  type: T_FLOAT
  input_op_param {
    shape {
      dim: 1
      dim: 3
      dim: 256
      dim: 256
    }
  }
}
node {
  name: "ssd_KYnet_v2/conv1_1/Conv2D"
  input: "placeholder_0"
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
node {
  name: "ssd_KYnet_v2/conv2_1/Conv2D"
  input: "ssd_KYnet_v2/conv1_1/Conv2D"
  op: "Conv2D"
  device: CPU
  type: T_FLOAT
  conv2d_op_param {
    depth: 32
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
```

### 2.3 优化SG
优化SG主要对模型进行数据位宽、精度和数据稀疏性分析。只需要指定plumber `sg` 命令编译后的文件夹路径，该文件夹路径包含了`sg`命令生成的所有文件，具体指令如下。
```shell
plumber_cli sg_opt -d ./tmp
```

用户也可以通过执行`./3_plumber_SG_opt.sh`来完成优化SG的步骤。

运行结果为：
```shell
Explored data representation table:
                                    node_name         key       min         max rep_fixed32 rep_ufixed32 rep_fixed16 rep_ufixed16 rep_fixed8 rep_ufixed8
0                                   img_input       ofmap  0.002121  254.997851       23, 8        24, 8        7, 8         8, 8       None        None
1                                   img_input       ifmap  0.002121  254.997851       23, 8        24, 8        7, 8         8, 8       None        None
2                 ssd_KYnet_v2/conv1_1/Conv2D        bias -0.179258    0.331259       30, 1        31, 1       14, 1        15, 1       6, 1        7, 1
3                 ssd_KYnet_v2/conv1_1/Conv2D       ifmap  0.002121  254.997851       23, 8        24, 8        7, 8         8, 8       None        None
4                 ssd_KYnet_v2/conv1_1/Conv2D     weights -0.621424    0.690765       30, 1        31, 1       14, 1        15, 1       6, 1        7, 1

```
```shell
2018-12-08 13:06:30 INFO Running optimization pass "Sparsity Optimisation" ...
                                    node_name      key      num      nnz  sparsity sparsity (%)
0                                   img_input    ofmap   196608        0  0.000000        0.00%
1                                   img_input    ifmap   196608        0  0.000000        0.00%
2                 ssd_KYnet_v2/conv1_1/Conv2D     bias       16        2  0.125000       12.50%
3                 ssd_KYnet_v2/conv1_1/Conv2D    ifmap   196608        0  0.000000        0.00%
4                 ssd_KYnet_v2/conv1_1/Conv2D  weights      432      216  0.500000       50.00%
5                 ssd_KYnet_v2/conv1_1/Conv2D    ofmap  1048576   547478  0.522116       52.21%
6                 ssd_KYnet_v2/conv2_1/Conv2D     bias       32       15  0.468750       46.88%
7                 ssd_KYnet_v2/conv2_1/Conv2D    ifmap  1048576   547478  0.522116       52.21%

```

### 2.4 为硬件优化SG
`-d tmp`同理上述三个步骤，都是是指定问件存放路径`tmp`。

`rainman_board_v3.pbtxt`是板卡参数定义文件，针对不同的板卡，硬件优化执行不同的优化操作，并分配不同的硬件执行设备。在步骤中会生成`model_hdl_sg.pbtxt`文件，它将作为`Raintime`的模型输入，具体指令如下：

```shell
plumber_cli hdl_opt ../rainman_board_v3.pbtxt -d /tmp
```

用户也可以通过执行`./4_plumber_HDL_opt.sh`来完成此步骤。


在此步骤中执行3个优化过程。

|步骤名称|功能|
|:-:|:-:|
|融合|将多个算子融合成一个|
|设备|使用执行设备标记SG节点|
|设计|设计空间探索|


### 2.5 导出数据
导出模型中的参数和每一个`SGNode`的输入和输出结果。
`-d tmp` 指定前面编译步骤生成的文件路径，`-b 16`指定导出定点数据的位宽，默认支持`8-bits, 16-bits, 32-bits`，根据你的模型数据来决定。在板卡上我们是使用定点数据来进行计算，所以需要知道数据位宽，同时还需指定数据精度，`-ib` 指定为整数位宽`9`位，`-fb`小数位宽`6`位，还剩一位为符号位。`--use-hdl=True`则指定使用硬件优化后的`SG`作为输入来导出数据。

导出的数据存放在`/app/tmp/data_hdl`中，里面有四个文件夹，分为：
  - 大端浮点数据存放在`float_big`文件
  - 大端定点数据存放在`fixed_big`文件
  - 小端浮点数据存放在`float_little`文件
  - 小端定点数据存放在`fixed_little`文件

请确保`filter_change.py`在docker的`/app`路径内。

本步骤的具体指令如下：
```shell
plumber_cli export -d tmp -b 16 -ib 8 -fb 7 --use-hdl True
python filter_change.py 32 4 tmp/data_hdl/float_little/
```

用户也可以通过执行`5_plumber_export_data.sh`脚本来完成此步骤。


### 2.6 将FPGA需要的数据整理到相同文件夹
将FPGA板卡中需要的文件整理到board_file中，FPGA板卡需要板卡文件为：`rainman9.6.rbf`、模型文件`*.pbtxt`、数据文件`float_little`以及后处理文件`post_params_46664`。其中后处理文件`post_params_46664`需要用户从我们提供的U盘中拷贝出来，该文件位于`/Tutorial03/文件/face/`下。


用户可以通过执行`6_cp_board_files.sh`脚本来完成此步骤，脚本中的具体指令为：

```shell
if [ ! -d ./board_files ];
then mkdir board_files
fi

cp -r post_params ./board_files/
cp tmp/model_hdl_sg.pbtxt ./board_files/
cp -r tmp/data_hdl/float_little/ ./board_files/
```
### 2.7 配置板卡并将需要使用的相关文件上传至板卡

#### 2.7.1 硬件设置与板卡登录

1. 将雨人v3加速器板放在平坦且不导电的表面上。
2. 将雨人加速器板上的以太网端口直接连接到主机PC的以太网端口。
3. 通过USB Type-C接口为加速器板供电。我们建议使用5V 2A手机充电器为电路板供电。 PC或笔记本电脑的USB端口可能无法提供所需的足够和稳定的电量。
4. 然后加速器板正常供电，板上（SD卡附近）的绿色LED亮起。如果使用带风扇进行散热的版本，正上电之后，风扇开启。



在主机系统中，**必须手动配置以太网到加速器板的网络接口IPv4地址，否则无法链接雨人V3板卡**，配置参数如下：

|设置|值|
|:---: |:---:|
|本地主机IPv4 | 192.168.123.10|
|网络掩码| 255.255.255.0|
|雨人V3板卡默认网关| 192.168.123.8|

#### 2.7.2 Docker文件向板卡的传输
1. Docker向虚拟机的文件传输

在传输文件之前，我们需要对板卡板卡执行文件进行编译。

首先我们清除历史编译记录，重建编译路径，进入`./workspace/16bit/single_img_tt3/build`文件夹，由于在文件拷贝过程中有会路径设置问题，优先清理`build`文件夹内的编译文件。

删除历史文件：
```shell
cd build
rm -r *
```

使用cmake构建文件和配置：
```shell
cmake ..
```

编译人脸检测程序：
```shell 
make
```
编译成功的结果如下，最后我们得到可一个可执行文件`demo_runner_face5b`：
```shell
Scanning dependencies of target demo_runner_face5b
[ 25%] Building CXX object CMakeFiles/demo_runner_face5b.dir/RainBuilder.cc.o
[ 50%] Building CXX object CMakeFiles/demo_runner_face5b.dir/post_process_tutorial.cc.o
[ 75%] Building CXX object CMakeFiles/demo_runner_face5b.dir/ssd_5b_runner.cc.o
[100%] Linking CXX executable demo_runner_face5b
[100%] Built target demo_runner_face5b
```

在Docker中，我们使用了plumber生成了包含了人脸检测神经网络相关硬件配置（pbtxt）和参数文件(float_little）。并且，板卡资源的调用需要通过可执行文件`demo_runner_face5b`进行。


因此，我们需要将docker中的描述文件`model_hdl_sg.pbtxt`、模型参数文件`float_little`和可执行`demo_runner_face5b`三个文件和拷贝至板卡中。

我们可以使用以下命令完成这这些文件从Docker向虚拟机`board_files`文件夹的拷贝过程。

在虚拟机中打开终端，输入以下命令：
```
sudo docker cp plumber_env_cpu_16bit:/app/board_files/model_hdl_sg.pbtxt /home/tutorial/board_files/
sudo docker cp plumber_env_cpu_16bit:/app/board_files/float_little /home/tutorial/board_files/
sudo docker cp plumber_env_cpu_16bit:/workspace/16bit/single_img_tt3/build/demo_runner_face5b /home/tutorial/board_files/
```

同时，后处理文件为U盘中的`post_param_46664`，放置在U盘中`Tutorial03/数据文件/face`路径下，并且用户需要提前下载好测试图片，为了方便，我们建议用户将后处理文件和测试图片放置在同一个文件`board_files`中。


1. 虚拟机向板卡的文件传输
   
通过`ssh`登录板卡后，可以使用以下命令将虚拟机上的文件拷贝至板卡中来。
```
scp -r toturial@{VM_IP}:/home/tutorial/board_files /workspace/
```

### 2.8 登录板卡并使用FPGA运行人脸检测程序



#### 2.8.1 板卡登录
使用以下命令登录板卡：
```shell
ssh root@192.168.123.8
```
输入密码：
```shell
letmein
```
通过SSH登录加速板后用户登录板卡后的默认路径为`root`，由于板卡环境默认为8bit，因此用户需要手动把路径切换到`/workspace/share`下。将板卡环境切换为16bit。

具体命令如下：
```
cd /workspace
./swithc_hw_rbf.sh 16     #切换板卡系统
```

完成切换后板卡会自动重启，因此需要用户使用`ssh`命令重新登录板卡，再回到`/workspace`路径下后，加载对应的板卡环境。命令如下：

```
./load_raintime16.sh      #加载16bit环境
```

检查许可证是使用雨人平台的软件源和硬件资源时所必需的步骤。**每次板卡重新启动，都需要在`/workspace`路径下执行该命令**。
```shell
./check_license
```


#### 2.8.2 运行人脸检测程序：

在`/workspace/share/board_files`路径下
运行人脸检测程序：
```shell
./demo_runner_face5b --pbtxt ./model_hdl_sg.pbtxt --coeff_path ./float_little  --param_path ./post_param_46664 --input_path ./timg.jpg --output_path ./out.jpg --cls 2
```

`--pbtxt`：FPGA硬件配置参数路径。
`--coeff_path`：数据路径。
`param_path`：后处理参数路径。
`input_path`：输入图片路径。
`output_path`: 输出图片路径。


在文件夹内得到图片计算结果`out.jpg`，示例如下：

<img src="https://i.loli.net/2018/12/08/5c0bd42182403.jpg" width=250 height=320 />


用户可以使用`scp`命令将输出图片从板卡上拷贝至虚拟机中进行查看。
```
scp /workspace/share/board_files/out.jpg tutorial@{VM_IP}:/home/tutorial/
```

## 练习1： 请使用行人检测模型对图片实现行人检测

本练习提供经过训练的行人检测算法模型checkpoint文件，请重复以上步骤完成行人检测算法部署。

模型文件在文件夹`/Tutorial03/文件/pedestrain/inference_model_pedestrian`下，包含`checkpoint`, `*.meta`,`*.index`, `*.data`。请用户将模型文件拷贝至docker的`/app`中。

由于行人检测算法与人脸检测算法在后处理上存在着差异，因此在编译时需要将`/workspace/16bit/single_ing_tt3`文件夹下的`Rainbuilder.cc`文件进行修改。

具体的修改内容为：
1. 通过vim命令进入.cc文件`vim RainBuilder.cc`。
2. 按下`i`进入编辑模式，将4，5，6行注释掉。
   ```
   //int num_anchors[NUM_EXY_LAYERS] = {4, 6, 6, 6, 4};
   //int feat_sizes[NUM_EXY_LAYERS] = {32 * 32 *4, 16 * 16 * 6, 8 * 8 * 6, 
                                        4 * 4 * 6, 2 * 2 * 4};
   ```
3. 将11，12，13行解除注释。
   ```
   int num_anchors[NUM_EXY_LAYERS] = {4, 4, 4, 4, 4};
   int feat_sizes[NUM_EXY_LAYERS] = {32 * 32 *4, 16 * 16 * 4, 8 * 8 * 4, 
                                        4 * 4 * 4, 2 * 2 * 4};
   ```
4. 按下`Esc`退出编辑模式，输入`:wq!`保存退出。
5. 进入`build`进行编译，具体操作同本教程2.7.2。


**预期结果如下：**

<img src="https://i.loli.net/2018/12/08/5c0bd66345089.jpg" width=250 height=320 />


## 目前检测算法后处理介绍
在以上的目标检测算法中，我们使用了一种叫NMS的算法消除了多次检测结果的干扰，本次实验将展示NMS对最后检测结果的影响。

**NMS原理：**

目标检测算法通常会输出很多检测框的结果，这些结果中有很多是冗余的，即有些框之间重叠区域很大因此可以进行合并。非极大值抑制（NMS）的目的就是要去除冗余的检测框,保留最好的一个。NMS实现的效果如下图：
 
<img src="https://i.loli.net/2018/12/08/5c0bd7fd9d8d6.png" width=400 height=320 />


**算法原理：**
1.	将所有框按照置信度从大到小排序，选中其中最高分值对应的框；
2.	遍历剩余的框，计算每个框与第一步选中框的IOU，如果大于一定阈值，将当前框删除。
3.	从剩余框中选出置信度最高的框，重复上述过程。

**NMS算法伪码：**

```shell
for i in n:
	for j in i+1 … n:
		If IOU(box[i], box[j])>threshold
			Delete box[j];
```

**IOU计算（Intersection over Union），计算两个框之间的交叠率。**


<img src="https://i.loli.net/2018/12/08/5c0be1960c291.png" width=400 height=320 />

```python
iou = interArea / (boxAArea + boxBArea - interArea)
```

`interArea`: 两个方框交错的面积。
`boxAArea + boxBArea`: 两个方框的面积和。


## 练习2：实现后处理并观测结果。

行人检测模型对应的后处理文件为`post_param_44444`，该文件与模型文件在`board_files`文件夹下，其中行人检测的`board_files`在U盘中的`Tutorial03/数据文件/pedestrain/`中。

将plumnber生成的模型描述文件`model_hdl_sg`、模型参数文件`float_little`、重新编译生成的可执行文件`demo_runner_face5b`拷贝到板卡`./workspace/share`文件夹下，再将后处理参数文件`post_param_44444`拷贝至同一路径下。并运行相应命令（形式如本教程2.8.2），可以得到下图的结果，可以看到，如果没有NMS，那么在检测结果图片中会出现了多个方框。接下来，我们要实现IOUT计算和NMS算法选取最优方框。

<img src="https://i.loli.net/2018/12/08/5c0bdab3b3e71.jpg" width=250 height=320 />


>**注意**：
>1. 请在`bbox.h`中查看方框的数据结构。
>2. 在docker中的`workspace/16bit/single_img_tt3/post_process_tutorial.cpp`文件中，已经预留了代码区域加入NMS后处理。
>3. `bbox1.volume()`表示box1的面积，`bbox1.volume()`表示box2的面积。


第一部分为计算IOU，我们已经给出了两个方框的坐标位置，请在以下代码中完成IOUT计算。
```cpp
float ComputeIOU(libssd::BBox &bbox1, libssd::BBox &bbox2) {
    float int_xmin = std::max(bbox1.xmin(), bbox2.xmin());
    float int_xmax = std::min(bbox1.xmax(), bbox2.xmax());
    float int_ymin = std::max(bbox1.ymin(), bbox2.ymin());
    float int_ymax = std::min(bbox1.ymax(), bbox2.ymax());
	
	// **********Insert your code at here*****************




  }
```
第二部分为NMS算法实现，在NMS计算中，需要调用IOU计算的结果：
```cpp
std::vector<libssd::BBox> NmsFunc(std::vector<libssd::BBox> &bboxes,
                                        float threshold) {
  std::vector<libssd::BBox> results;
  std::vector<bool> keep(bboxes.size(), true);
  
  // sort all bounding boxes by score
  std::sort(bboxes.rbegin(), bboxes.rend());

  for (int i = 0; i < bboxes.size(); i++) {
    if (keep[i]) {
      results.push_back(bboxes[i]);
      for (int j = i + 1; j < bboxes.size(); j++) {
      // **********Insert your code at here*****************
        
                   
                  
      }
    }
  }

  return results;
}
```

编辑`post_process_tutorial.cpp`完成后，在`build`文件夹中进行编译。运行`run_ped5b.sh`进行验证。如果得到只有一个方框的输出图片，则得到正确结果。

实验结束！