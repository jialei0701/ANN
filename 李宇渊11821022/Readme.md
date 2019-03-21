### About me
* **Name**: 李宇渊  
* **Student ID**: 11821022  
* **Topic**: Neural Networks
### Schedule

Task|Due|Done
-|:-:|:-:
1.选择论文|Mar.14|T
2.精读论文|Mar.21|T
3.复现论文|Apr.4|
4.完成实验|Apr.11|
5.撰写报告|Apr.18|  
### 选择论文
[FNText: A Fast Neural Model for Efficient Text Classification](Fntext.pdf)  
这篇论文利用神经网络进行文本分类。传统方法大多利用CNN和RNN等结构搭建深度神经网络，这篇论文构建了一个简单高效的三层神经网络。该模型训练时间短，计算资源要求低，准确率也较为出色。  
* **Abstract**
>In recent years,very deep neural models based convolutional neural networks (CNNs) have achieved remarkable results in natural language processing (NLP). However, the computational complexity also largely increasesas the networks go deeper, which causes long training time. To raise the efﬁciency of calculation, this paper focus on shallow neural model and explores a fast neural text classiﬁcation model FNText, which only contains 3 layers,without activation function and stacked time-consuming convolutional layers. Instead of enumerating a bag of bi-grams, we propose a novel method which utilizes average pooling operation along randomly initializing word vectors to obtain bi-gram features. These additional bi-gramfeatures can further improve the performance of FNText. We improve the training speed by ignoring hyperparameters with zero-gradients. Experiments show that FNText can be trained on more than 300 million words in less than 10 minutes using a standard multicore CPU, and achieves competitive results on several large-scale datasets. Sometimes FNText is on par with very deep neural models.
### 精读论文
#### 中心思想
* 就测试准确率而言，该模型可以达到甚至超越深度网络模型；
* 该模型层数少，无激活函数，训练速度快，可以在一个标准多核CPU上训练；
* 该模型无RNN等循环结构，可以并行训练。
#### 模型结构
论文中提出了该模型的两种版本，without bi-gram和with bi-gram。without bi-gram版本的模型已经可以取得不错的效果，利用了bi-gram以后，模型的效果更加理想。其中利用了bi-gram的具体结构如下如所示：  
![bi_stru](bi_stru.JPG)  
该结构对每个词随机产生一个词向量，然后对词向量的每个维度进行max pooling，最后输入一层全连接神经网络进行softmax分类。
#### 训练方法
Adam
#### 实验结果
作者在多个数据集上进行了对比实验，结果如下图所示：
![test_res](test_res.JPG)
### 复现论文

### 完成实验

### 撰写报告
