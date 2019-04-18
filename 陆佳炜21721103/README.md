## 生物智能算法 神经网络组
#### 姓名：陆佳炜 
#### 学号：21721103
#### 邮箱：jarvis_lu@zju.edu.cn

---

## Timeine  

| Task | Due | Done |
| :- | :- | :- |
| 1. 选择综述论文 | Mar. 14 | &radic; |  
| 2. 精读论文，理解模型 | Mar. 21 | &radic; |  
| 3. 复现论文 | Apr. 4 | &radic; |  
| 4. 完成对比实验 | Apr. 11 | &radic; |  
| 5. 形成最后报告 | Apr. 18 | &radic; |  

### 1. 选择论文

[Dance with Melody: An LSTM-autoencoder Approach toMusic-oriented Dance Synthesis](https://hcsi.cs.tsinghua.edu.cn/Paper/Paper18/MM18-TANGTAORAN.pdf)
      
    主题： 基于LSTM-Autoencoder的方法进行面向音乐的舞蹈动作生成
    论文： https://hcsi.cs.tsinghua.edu.cn/Paper/Paper18/MM18-TANGTAORAN.pdf
    该论文获得了 ACM MultiMedia 2018 Best Demo 的奖项


<div align=center><img src="./resources/demo.png" /></div>


    舞蹈和音乐具有很强的相关性。关于如何合成面向音乐的舞蹈的研究可以促进很多领域的发展，例如：舞蹈教学、人类行为学研究。
    
    本文构建了一个基于LSTM长短时记忆网络和自编码器网络的智能系统，可以提取出从音乐特征到舞蹈特征的映射关系。
    并提出了加入时序索引特征和节奏筛子的方式，得到了更好的效果。
    
    此外，该工作还建立了音乐-舞蹈的数据集，其中含有4种类型舞蹈的907,200帧3D舞蹈动作数据和对应的音乐数据。

----

### 2. 精读论文, 理解模型

- #### 2.1 模型架构

      LSTM网络在编码序列数据的Case中表现优秀，而且相比于RNN，LSTM在长期的预测上更加鲁棒。

      舞蹈动作序列也是序列的一种，因此本文使用LSTM网络作为基础来对问题进行建模。

      文中提到naive的做法是使用单一的LSTM网络进行建模，如下图所示：

     <div align=center><img src="./resources/naive_approach.png" /></div>

      该方法对音乐特征编码，加窗，每个时间步，输入一个音乐特征，通过LSTM单元输出一个动作特征。
        
     <div align=center><img src="./resources/final_approach.png" /></div>


#### 2.2 数据集

[Dance Dataset](https://github.com/Jarvisss/Music-to-Dance-Motion-Synthesis)

#### 2.3 特征提取



#### 2.4 细节阐述


### 3. 复现论文和改进

[*Todo*]

### 4. 对比实验

[*Todo*]

### 5. 最终报告

[*Todo*]
