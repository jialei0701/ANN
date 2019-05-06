### Personal information
* **Name**: 朱婕  
* **Student ID**: 21821291  
* **Topic**: Neural Networks
### Schedule
Task|Due|Done
-|:-:|:-:
1.选择论文|Mar.14|T
2.精读论文|Mar.21|T
3.复现论文|Apr.4|T
4.完成实验|Apr.11|T
5.撰写报告|Apr.18|T
### 选择论文
Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
* **Abstract**
>图（graph）是一种数据格式，它可以用于表示社交网络、通信网络、蛋白分子网络等，图中的节点表示网络中的个体，连边表示个体之间的连接关系。许多机器学习任务例如社团发现、链路预测等都需要用到图结构数据，因此图卷积神经网络的出现为这些问题的解决提供了新的思路。
作者提出了一种可扩展的方法，用于图结构数据的半监督学习，该方法基于一个可直接在图上操作的卷积神经网络的有效变体。卷积结构的选择来源于频谱图卷积的局部一阶近似来激励。此模型在图中边的数量上线性缩放，并学习隐藏层表示，其编码局部图结构和节点的特征。此文提出的GCN（图卷积网络）为图结构数据的处理提供了一个崭新的思路，将深度学习中常用于图像的卷积神经网络应用到图数据上。
### 精读论文
[详见精读论文.pdf](https://github.com/jialei0701/ANN/blob/master/%E6%9C%B1%E5%A9%9521821291/%E7%B2%BE%E8%AF%BB%E8%AE%BA%E6%96%87.pdf)
### 复现论文
[详见gcn](https://github.com/jialei0701/ANN/tree/master/%E6%9C%B1%E5%A9%9521821291/gcn)
代码在gcn文件夹中，在cora数据集上的运行结果如下图所示，cora是文献引用网络，共六种类别。图中loss表示交叉熵损失，acc表示准确率，模型的训练只使用了引用网络中150个节点的标签，比例是0.052，可以看出模型的半监督学习效果良好：
![模型训练结果](https://github.com/jialei0701/ANN/blob/master/%E6%9C%B1%E5%A9%9521821291/train.png)
### 完成实验
**模型隐藏层输出降维可视化**
![模型隐藏层输出降维可视化](https://github.com/jialei0701/ANN/blob/master/%E6%9C%B1%E5%A9%9521821291/tSNE1.PNG)  
-------
**损失函数（交叉熵）可视化**
![损失函数（交叉熵）可视化](https://github.com/jialei0701/ANN/blob/master/%E6%9C%B1%E5%A9%9521821291/3d_entropy.png)
### 撰写报告
这篇文章介绍了一种基于图结构数据的半监督分类方法。GCN模型使用了一个有效的层级传播规则，该规则基于图上谱卷积的一阶近似，有效的利用了一个节点的领域信息。实验结果表明，该GCN模型能够同时编码图结构和节点特征，对半监督分类有一定帮助。
[详见实验报告.pdf](https://github.com/jialei0701/ANN/blob/master/%E6%9C%B1%E5%A9%9521821291/%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.pdf)
