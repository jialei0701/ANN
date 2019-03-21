### About me
* **Name**: 徐帅  
* **Student ID**: 21821174  
* **Email**: xushuai100@gmail.com
### Schedule

Task|Due|Done
-|:-:|:-:
1.选择论文|Mar.14|T
2.精读论文|Mar.21|T
3.复现论文|Apr.4|
4.完成实验|Apr.11|
5.撰写报告|Apr.18|  
### 选择论文
[[IJCAI17]deep matrix factorization models for recommender systems](DMF.pdf)
>作者将基于矩阵分解(Matrix Factorization)的推荐系统以输入为标准分为了两类，一种是基于隐式反馈(implicit feedback)为输入的推荐系统，另一种则是基于显式反馈(explicit feedback)为输入的推荐系统。隐式反馈输入往往是无法表达用户具体情感的数据，最常见的表达形式就是0-1矩阵，比如用户对物品的浏览情况，可以通过以用户为行，物品为列，构造一个0-1矩阵表达用户是否与物品存在交互。而显式反馈输入则往往比较明显了量化了用户的情感，最常见的就是评分矩阵，这种矩阵是通过以用户为行，物品为列，矩阵的元素为评分的形式构造。光从数字上看，一般隐式反馈比较适合与做提高召回率(recall)的模型输入。

由于推荐系统的数据集通常都是关于评分的，因此有一种比较特殊的隐式反馈输入，即将评分全都视为1，而这种变化会引起信息丢失的问题。
在这篇文章中，作者通过结合显式反馈和隐式反馈这两种输入，有效的解决了信息丢失的问题。不仅如此，作者还通过结合神经网络，提出了一种新型的矩阵分解模型，称为Deep Matrix Factorization (DMF)
* **Abstract**

## 2.精读论文
#### 数据集: ML100k ML1m Amovie Amusic
#### 模型架构
![model Architecture](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/arch.png)
1. 初始化两个神经网络
2. 将行、列向量 Y_{i,*} ， Y_{*,j}  分别输入到两个神经网络中，一层层进行特征提取，并且确保两个神经网络输出特征的维度一样，此时输出的就是用户特征 p_i 和物品特征  q_j 。
3. 计算两个特征的相似度，相似度越高，那么用户i越有可能对物品j感兴趣，作者选用余弦相似度描述两者的相似程度：  ![cos_equ](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/cos.svg)
#### 损失函数
得到的Y_{ij}便是预测的概率值 ，这篇文章的一个亮点是其中损失函数的构造？

作者首先提出了一个general function:

![loss1](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss1.svg)

其中 l(\cdot) 描述的是误差， \Omega(\cdot) 描述的是正则化项

和大多数论文一样，作者将主要精力用于寻找一个合适的 l(\cdot) 。

作者首先想到的是平方误差，即

![loss2](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss2.svg)

但是由于 \hat{Y}_{ij} 是一个预测的概率，并不适用于显式反馈（评分）。因此和大多数机器学习方法一样，作者选用了信息论中的交叉熵(cross-entropy)来描述误差。最原始的形式如下：

![loss3](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss3.svg)

为了能让上述式子满足交叉熵的定义， Y_{ij} 必须是隐式反馈，但是为了满足论文初衷，作者认为显式反馈会得到更好的表达, 因此为了能够引入显式反馈，作者将评分归一化，得到如下损失函数:

![loss4](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss4.svg)

上面的式子不再是纯粹的交叉熵，归一化的评分可以理解为权重，评分越高的误分类到0的惩罚越高，反之，评分越低无分类到1的惩罚也越高，这符合我们的认知，因此该转化是合理的。

得到误差之后，便可以分别反向传播回神经网络，用以更新权重信息
#### 结果
![result](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/result.png)
#### 总结
总的来说，该文章的贡献如下：
1. 利用DMF将显示反馈输入中包含的用户和物品信息非线性地映射到了一个低维空间
2. 提出一种既包含显式反馈又包含隐式反馈的损失函数
3. 在多个数据集上跑出来的效果都非常可观
### 复现论文

### 完成实验

### 撰写报告
