# 生物智能算法 神经网络组

## Personal information
+ Name: 贾磊
+ Student ID: 11816008
+ Email: jialei0701@foxmail.com

---

## Timeline

|Task|Date|Done|
--|--|:--:
1.选择论文|Mar. 14|√
2.精读论文，理解模型|Mar. 21|√
3.复现论文|Mar. 28|
4.完成对比实验|Apr. 4|
5.形成报告|Apr. 11|

---

## 1. 选择论文

**Title:**

[CNNsite: Prediction of DNA-binding Residues in Proteins Using Convolutional Neural Network with Sequence Features.](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/Zhou%20et%20al.%20-%202017%20-%20CNNsite%20Prediction%20of%20DNA-binding%20residues%20in%20proteins%20using%20Convolutional%20Neural%20Network%20with%20sequence%20features.pdf)

>IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2016

**Abstract:**

>Protein-DNA complexes play crucial roles in gene regulation. The prediction of the residues involved in protein-DNA interactions is critical for understanding gene regulation. Although many methods have been proposed, most of them overlooked motif features. Motif features are sub sequences and are important for the recognition between a protein and DNA. In order to efficiently use motif features for the prediction of DNA-binding residues, we first apply the Convolutional Neural Network (CNN) method to capture the motif features from the sequences around the target residues. CNN modeling consists of a set of learnable motif detectors that can capture the important motif features by scanning the sequences around the target residues. Then we use a neural network classifier, referred to as CNNsite, by combining the captured motif features, sequence features and evolutionary features to predict binding residues from sequences.

**摘要**
>蛋白-DNA复合体在基因调控的过程中扮演着重要的作用。对参与到蛋白-DNA互作的残基（residues）的预测对于理解基因调控有重要意义。现在已经有一些预测方法，但是这些方法忽视了基序（motif）的特征。基序特征是亚序列，其对蛋白质和DNA的识别具有重要意义。为了有效利用基序特征进行DNA绑定残基的鉴定，本研究应用卷积神经网络来提取目标残基周围序列的基序体征。

---

## 2. 精读论文，理解模型

**数据集**

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/datasets.jpg)

**Framework**

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/framework.jpg)

**Convolution layer**

&emsp; 输入residue-wise数据S左右填补（m-1）的unuseful residue，转换为矩阵M（类图像像素数据）；

&emsp; 输出为矩阵X，其中X<sub>i,k</sub>表示第k个motif detector在第i个位置的得分；

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/conv_layer.jpg)


**Rectification layer**

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/rectification_layer.jpg)

&emsp; 过滤非高效motif特征

**Pooling layer**

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/pooling_layer.jpg)

&emsp;最大池化

**Neural network layer**

&emsp; 综合motif特征、sequence特征、evolutionary特征进行预测。
&emsp; 采用dropout technique避免overfitting。


**不同特征比较**

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/ROC.jpg)

**方法间比较**

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/compare.jpg)

&emsp; Sensitivity (SN), Specificity (SP), Strength (ST), Accuracy (ACC), and Mathews Correlation Coefficient (MCC).

---
## 3. 复现论文

---
## 4. 完成对比实验

---
## 5. 形成报告


