### 生物智能与算法

#### Timeline  

| Task | Due | Done |
| --- | --- | :-: |
| 1. 选择综述论文 | Mar. 14 |  |  
| 2. 精读论文，理解模型 | Mar. 21 |  |  
| 3. 复现论文 | Apr. 4 |  |  
| 4. 完成对比实验 | Apr. 11 |  |  
| 5. 形成最后报告 | Apr. 18 |  |  

#### Mar. 14  
选择推荐系统领域的两篇论文进行综述，分别为
> Man T, Shen H, Jin X, et al. Cross-Domain Recommendation: An Embedding and Mapping Approach[C]//IJCAI. 2017: 2464-2470.

> He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017: 173-182.

其中第一篇论文从嵌入和映射角度对跨域推荐系统进行研究，即在不同域之间学习一个域间映射函数，继而解决跨域推荐中存在的数据异构问题。
本文提出的EMCDR框架与现有的跨域推荐系统存在两方面的不同：首先，使用多层神经网络实现跨域的非线性映射，非线性映射为单域实体提供了更灵活的学习算法。其次，只有拥有数据充分的实体被用来学习映射函数，因此该模型对单域数据中噪声具有鲁棒性。
简单来说，EMCDR可以分为三个部分，首先是分别对不同域的用户和物品进行特征建模，其次使用神经网络对两个域间共享特征进行映射，最后实现跨域推荐。

第二篇论文针对协同过滤推荐中的经典算法——概率矩阵分解(Probabilistic Matrix Factorization)提出改进的深度神经协同过滤(NCF)。本文认为，传统的矩阵分解方法使用线性方法(点乘)来描述用户和物品之间的交互关系，这是不够充分的。因此提出了使用神经网络来学习用户-物品的交互函数，从而实现用户特征和物品特征之间的非线性组合。

本次课程探究我将采用EMCDR框架，并借鉴NCF思想，改进EMCDR框架，以此探究深度学习在跨域推荐系统中的作用。
