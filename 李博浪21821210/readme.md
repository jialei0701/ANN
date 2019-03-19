### About me
* **Name**: 李博浪
* **Student ID**: 218212100  
* **Topic**: Neural Networks

### Schedule

| Task | Due | Done |
| :--:| :--: | :--: |
| 1.选择论文 | Mar.14 | Y |
| 2.精读论文 | Mar.21 | N
| 3.复现论文 | Apr.4 | N
| 4. 完成实验 | Apr.11 | N
| 5.撰写报告 | Apr.18 | N

### 选择论文
[Deep Neural Networks for Learning Graph Representations](DNGR.pdf)  

* **摘要**
> &ensp;本文提出了一种新的模型,通过捕获图的结构特征,从而能够学习图的表示,为图中每个节点生成低维向量.与以往的基于采样的模型不同,DNGR主要通过随机搜索从而直接捕获图的结构信息.本文主要使用堆叠去噪自编码器去抽取PMI矩阵中的复杂的非线性特征.为了证明模型的有效性,作者利用了模型所学习到的节点的向量,来完成聚类及可视化任务等.
### 精读论文
>本文通过捕获图的特征结构来学习图的表示,为图中每个节点生成低维向量.本文可用于带权图,而且还能捕获图中的非线性关系.  
本文算法主要分为三个步骤:  
&emsp;1.使用random surfing模型捕获图的结构,并获得过线概率矩阵PCO  
&emsp;2.基于PCO来计算PPMI矩阵  
&emsp;3.利用堆叠去噪自编码器来学习节点的低维表示  
1.random surfing  
&emsp;$p_{k}=\alpha p_{k-1}A+(1-\alpha )p_{0}$  
&emsp;$r=\sum_{k=1}^{K}p_{k}$  
即通过转移概率矩阵来求得节点的k跳所能到达节点的概率,然后通过加权和获得节点的概率共现矩阵PCO.  
2.基于PCO来计算PPMI矩阵  
&emsp;$PMI_{w,c}=log(\frac{\#(w,c)\cdot \left | D \right |)}{\#(w)\cdot \#(c)}))$  
&emsp;$PPMI_{w,c}=max(PMI_{w,c},0)$  
通过节点的概率共现矩阵获得PPMI矩阵  
3.利用堆叠去噪自编码器来学习节点的低维表示  
将PPMI矩阵中节点对应的向量作为输入,放入SDAE里,获取节点的低维向量表示.并使用逐层的贪婪预训练.最终获得节点的低维向量表示


### 复现论文

### 完成实验

### 撰写报告
