## 生物智能算法 神经网络组
#### 姓名：江彦开 
#### 学号：11821017
#### 邮箱：jyk1996ver@zju.edu.cn

---

## Timeine  

| Task | Due | Done |
| :- | :- | :- |
| 1. 选择综述论文 | Mar. 14 | &radic; |  
| 2. 精读论文，理解模型 | Mar. 21 | &radic; |  
| 3. 复现论文 | Apr. 4 | &radic; |  
| 4. 完成对比实验 | Apr. 11 | &radic; |  
| 5. 形成最后报告 | Apr. 18 | &radic; |  


#### 1. 选择论文
[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
* **Abstract**
>提出了一个概念上简单，灵活和通用的目标分割框架。方法有效地检测图像中的目标，同时为每个实例生成高质量的分割掩码。称为Mask R-CNN的方法通过添加一个与现有目标检测框回归并行的，用于预测目标掩码的分支来扩展Faster R-CNN。Mask R-CNN训练简单，相对于Faster R-CNN，只需增加一个较小的开销，运行速度可达5 FPS。此外，Mask R-CNN很容易推广到其他任务，例如，允许在同一个框架中估计人的姿势。在COCO挑战的所有三个项目中取得了最佳成绩，包括目标分割，目标检测和人体关键点检测。在没有使用额外技巧的情况下，Mask R-CNN优于所有现有的单一模型，包括COCO 2016挑战优胜者。这种简单且有效的方法将成为一个促进未来目标级识别领域研究的坚实基础。


#### 2. 精读论文，理解模型
>Mask RCNN可以看做是一个通用实例分割架构。Mask RCNN以Faster RCNN原型，增加了一个分支用于分割任务。Mask RCNN比Faster RCNN速度慢一些，达到了5fps。

![img](https://github.com/jialei0701/ANN/blob/master/%E6%B1%9F%E5%BD%A6%E5%BC%8011821017/maskrcnn-image/20181017160239157.png)

>Mask-RCNN 的几个特点：

1）在边框识别的基础上添加分支网络，用于 语义Mask 识别；

2）训练简单，相对于 Faster 仅增加一个小的 Overhead，可以跑到 5FPS；

3）可以方便的扩展到其他任务，比如人的姿态估计 等；

4）不借助 Trick，在每个任务上，效果优于目前所有的 single-model entries(包括 COCO 2016 的Winners）。

>Mask-RCNN 技术要点

● 技术要点1 - 强化的基础网络

     通过 ResNeXt-101+FPN 用作特征提取网络，达到 state-of-the-art 的效果。

● 技术要点2 - ROIAlign

     采用 ROIAlign 替代 RoiPooling（改进池化操作）。引入了一个插值过程，先通过双线性插值到14*14，再 pooling到7*7，很大程度上解决了仅通过 Pooling 直接采样带来的 Misalignment 对齐问题。虽然 Misalignment 在分类问题上影响并不大，但在 Pixel 级别的 Mask 上会存在较大误差。后面我们把结果对比贴出来（Table2 c & d），能够看到 ROIAlign 带来较大的改进，可以看到，Stride 越大改进越明显。 
     
● 技术要点3 - Loss Function

     每个 ROIAlign 对应 K * m^2 维度的输出。K 对应类别个数，即输出 K 个mask，m对应 池化分辨率（7*7）。Loss 函数定义：
     Lmask(Cls_k) = Sigmoid (Cls_k)，    平均二值交叉熵 （average binary cross-entropy）Loss，通过逐像素的 Sigmoid 计算得到。
     Why K个mask？通过对每个 Class 对应一个 Mask 可以有效避免类间竞争
     
- Mask R-CNN基本结构：
     与Faster RCNN采用了相同的two-state步骤：首先是找出RPN，然后对RPN找到的每个RoI进行分类、定位、并找到binary mask。这与当时其他先找到mask然后在进行分类的网络是不同的。Mask Representation因为没有采用全连接层并且使用了RoIAlign，可以实现输出与输入的像素一一对应。
     
     RoIAlign：RoIPool的目的是为了从RPN网络确定的ROI中导出较小的特征图(a small feature map，eg 7x7)，ROI的大小各不相同，但是RoIPool后都变成了7x7大小。RPN网络会提出若干RoI的坐标以[x,y,w,h]表示，然后输入RoI Pooling，输出7x7大小的特征图供分类和定位使用。问题就出在RoI Pooling的输出大小是7x7上，如果RON网络输出的RoI大小是8*8的，那么无法保证输入像素和输出像素是一一对应，首先他们包含的信息量不同（有的是1对1，有的是1对2），其次他们的坐标无法和输入对应起来（1对2的那个RoI输出像素该对应哪个输入像素的坐标？）。这对分类没什么影响，但是对分割却影响很大。RoIAlign的输出坐标使用插值算法得到，不再量化；每个grid中的值也不再使用max，同样使用差值算法。
     
     ![img](https://github.com/jialei0701/ANN/blob/master/%E6%B1%9F%E5%BD%A6%E5%BC%8011821017/maskrcnn-image/2.png)
     
#### 3. 复现论文

- Network Architecture：为了表述清晰，有两种分类方法
    1. 使用了不同的backbone：resnet-50，resnet-101，resnext-50，resnext-101；

    2. 使用了不同的head Architecture：Faster RCNN使用resnet50时，从CONV4导出特征供RPN使用，这种叫做ResNet-50-C4

    3. 作者使用除了使用上述这些结构外，还使用了一种更加高效的backbone——FPN
    
        ![img](https://github.com/jialei0701/ANN/blob/master/%E6%B1%9F%E5%BD%A6%E5%BC%8011821017/maskrcnn-image/3.png)

- Implementation Details：
使用Fast/Faster相同的超参数，同样适用于Mask RCNN

● Training:
1、与之前相同，当IoU与Ground Truth的IoU大于0.5时才会被认为有效的RoI，只把有效RoI计算进去。

2、采用image-centric training，图像短边resize到800，每个GPU的mini-batch设置为2，每个图像生成N个RoI，对于backbone的N=64，对于FPN作为backbone的，N=512。作者服务器中使用了8块GPU，所以总的minibatch是16，迭代了160k次，初始lr=0.02，在迭代到120k次时，将lr设定到 lr=0.002，另外学习率的weight_decay=0.0001，momentum = 0.9。如果是resnext，初始lr=0.01,每个GPU的mini-batch是1。

3、RPN的anchors有5种scale，3种ratios。为了方便剥离、如果没有特别指出，则RPN网络是单独训练的且不与Mask RCNN共享权重。但是在本论文中，RPN和Mask R-CNN使用一个backbone，所以他们的权重是共享的。Ablation Experiments 为了方便研究整个网络中哪个部分其的作用到底有多大，需要把各部分剥离开。

#### 4. Experiment

[*Todo*]

#### 5. Final Report

[*Todo*]
