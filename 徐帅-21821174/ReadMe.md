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
2. 将行、列向量 ![row_vector](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/row_vector.svg) ， ![colum_vec](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/colum_vec.svg)  分别输入到两个神经网络中，一层层进行特征提取，并且确保两个神经网络输出特征的维度一样，此时输出的就是用户特征 p_i 和物品特征  q_j 。
3. 计算两个特征的相似度，相似度越高，那么用户i越有可能对物品j感兴趣，作者选用余弦相似度描述两者的相似程度：  ![cos_equ](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/cos.svg)
#### 损失函数
得到的![pred](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/pred.svg)便是预测的概率值 ，这篇文章的一个亮点是其中损失函数的构造

作者首先提出了一个general function:

![loss1](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss1.svg)

其中![loss](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss.svg) 描述的是误差， ![norm](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/norm.svg) 描述的是正则化项

和大多数论文一样，作者将主要精力用于寻找一个合适的 ![norm](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/norm.svg) 。

作者首先想到的是平方误差，即

![loss2](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss2.svg)

但是由于 ![hat_y](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/hat_y.svg) 是一个预测的概率，并不适用于显式反馈（评分）。因此和大多数机器学习方法一样，作者选用了信息论中的交叉熵(cross-entropy)来描述误差。最原始的形式如下：

![loss3](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss3.svg)

为了能让上述式子满足交叉熵的定义， ![pred](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/pred.svg) 必须是隐式  反馈，但是为了满足论文初衷，作者认为显式反馈会得到更好的表达, 因此为了能够引入显式反馈，作者将评分归一化，得到如下损失函数:

![loss4](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss4.svg)

上面的式子不再是纯粹的交叉熵，归一化的评分可以理解为权重，评分越高的误分类到0的惩罚越高，反之，评分越低无分类到1的惩罚也越高，这符合我们的认知，因此该转化是合理的。

得到误差之后，便可以分别反向传播回神经网络，用以更新权重信息

#### 总结
总的来说，该文章的贡献如下：
1. 利用DMF将显示反馈输入中包含的用户和物品信息非线性地映射到了一个低维空间
2. 提出一种既包含显式反馈又包含隐式反馈的损失函数
3. 在多个数据集上跑出来的效果都非常可观
### 复现论文
LFM虽然可以有效利用隐式反馈来学习到用户和物品的隐语义表示；然而LFM学习到的往往是一种线性的浅层的特征。基于此，我们利用多层感知机来获得用户和物品更深层次的特征，提出了多层隐语义模型（multi- latent factor model，MLFM）

**MLFM架构设想**

![MLFM-arch](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/MLFM.png)

我们将 LFM 算法以及 MLFM 算法进行融合，得到一种结合了用户与物品线性以及非线性特征的混合隐语义模型（fusion latent factor model，FLFM）

**FLFM架构设想**

![FLFM-arch](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/FLFM.png)

### 模型仿真验证

数据集|用户数目|物品数目|评分数目|数据稀疏度
-|:-:|:-:|:-:|:-:
Movielens|6040|3952|1000209|95.81% 
Pinterest|55187|9916|1500809|99.73% 

![LOSS](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/LossComp.png)

#### Loss函数可视化
交叉熵：
![loss3](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/loss3.svg)

损失函数直观图像：
![crossentropy](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/crossentropy2.png)

损失函数三维仿真图像：
![3Dcrossentropy](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/3D-cross-Entropy.jpg)

神经网络的损失函数很多都是非凸的，但可以证明，在本实验中，如果代价函数是交叉熵，最后一层是逻辑单元(sigmod函数），其他层是relu激活函数；那么此时的神经网络的损失函数是凸的。

假设函数f二阶可微，函数f是凸函数的充要条件是：其Hessian矩阵是半正定阵， 即![hessian](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/hessian.png)

不严谨推导如下：
![convex](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/convex.jpg)

### 完成实验
#### 评价标准
本次top-N推荐的性能由命中率（hit ration，HR）以及归一化折损累积增益（normalized discounted cumulative gain, NDCG）来进行衡量。 其中HR是一种召回率相关的指标，假设我们取推荐排名前K的物品作为推荐列表，那么HR就直观地衡量用户下一个交互的物品是否出现在推荐列表中。
其公式为： 

![HR](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/HR.png)

其中|GT|是所有的测试集合，而分子则是每个用户前K个中属于测试集合的个数的总和。 

而NDCG则是一种准确率相关的指标，衡量的是排序的质量。直观的来讲，命中物品在推荐列表中所处的位置越靠前，那么这一次推荐获得的分数也越高。
其计算公式为：

![NDCG](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/NDCG.png)

#### 实验结果
![result-16](https://github.com/jialei0701/ANN/blob/master/%E5%BE%90%E5%B8%85-21821174/result-16.png)

### 撰写报告
