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
>Mask RCNN可以看做是一个通用实例分割架构。
>Mask RCNN以Faster RCNN原型，增加了一个分支用于分割任务。
>Mask RCNN比Faster RCNN速度慢一些，达到了5fps。
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
#### 3. Implementation

[*Todo*]

#### 4. Experiment

[*Todo*]

#### 5. Final Report

[*Todo*]
