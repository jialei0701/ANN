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


      
> 主题： 基于LSTM-Autoencoder的方法进行面向音乐的舞蹈动作生成
>
> 论文： [Dance with Melody: An LSTM-autoencoder Approach toMusic-oriented Dance Synthesis](https://hcsi.cs.tsinghua.edu.cn/Paper/Paper18/MM18-TANGTAORAN.pdf)
>
> 该论文获得了 ACM MultiMedia 2018 Best Demo 的奖项


<div align=center><img src="./resources/demo.png" /></div>


> 舞蹈和音乐具有很强的相关性。
>
> 关于如何合成面向音乐的舞蹈的研究可以促进很多领域的发展，例如：舞蹈教学、人类行为学研究。本文构建了一个基于LSTM长短时记忆网络和自编码器网络的智能系统，可以提取出从音乐特征到舞蹈特征的映射关系。并提出了加入时序索引特征和节奏筛子的方式，得到了更好的效果。
>
> 此外，该工作还建立了音乐-舞蹈的数据集，其中含有4种类型舞蹈的907,200帧3D舞蹈动作数据和对应的音乐数据。

----

### 2. 精读论文, 理解模型

#### 2.1 模型架构

> LSTM网络是传统RNN的一种改进，在编码序列数据的案例中表现优秀。相比于Vanilla RNN，LSTM在长期的预测上更加鲁棒，因为一定程度上解决了RNN的梯度消失和梯度爆炸的问题，关于LSTM可以查看[这篇文章](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。
>
> 舞蹈动作序列也是序列的一种，因此本文使用LSTM单元作为基础来对问题进行建模。
>
> 文中提到naive的做法是使用单一的LSTM网络进行建模，如下图所示：

   <div align=center><img src="./resources/naive_approach.png" width="50%" height="50%"/></div>
     
> 该方法通过计算预测值和实际值的欧几里得距离，用该损失函数来进行模型的优化。

 <div align=center><img src="./resources/loss1.png" width="30%" height="30%"/></div>
 
> 首先对音乐进行加窗提取特征。每个时间步输入一个音乐特征(16维向量)，通过LSTM单元，根据当前隐层状态C和H，输出一个动作特征(69维向量)，同时改变LSTM网络的隐层状态C和H，输入到下一个时间步，进行下一步的预测。
> 
> 但是这种做法的问题是：**模型难以收敛**、预测的结果无法保证是否**在一个节拍内连续**。
>
> 因此作者提出了如下的**改进模型**：
        
   <div align=center><img src="./resources/final_approach.png" /></div>
   
> 在改进模型中，作者加入了音乐的Auto-Encoder模块，以进一步提取和处理音乐特征。

> 在这个模型中，损失函数又加入了Loss_extr, 该函数作用是使得压缩后的音乐特征尽量保持原来的信息。

 <div align=center><img src="./resources/loss2.png" width="30%" height="30%"/></div>
 
> 最终loss为:

 <div align=center><img src="./resources/loss.png" width="50%" height="50%"/></div>
 
> 损失函数仿真：

 <div align=center><img src="./resources/Figure_5.png" width="50%" height="50%"/></div>

#### 2.2 数据集

> 数据集为原作者通过动作捕捉设备采集的舞蹈数据，链接：[Dance Dataset](https://github.com/Jarvisss/Music-to-Dance-Motion-Synthesis)


#### 2.3 特征提取

   <div align=center><img src="./resources/acoustic.png" /></div>

> 对齐后的music使用librosa提取features, 对每个时间窗口提取16维的特征。

> a)	Mfcc[3], mfcc_delta[3] 人声
>
> b)	Cqt_chroma[4] 音调
>
> c)	Onset_envelope[5] 音量
>
> d)	Tempogram[1] 节拍周期

   <div align=center><img src="./resources/temporal.png" /></div>  

> a)    使用librosa.beat.beat_track()函数计算beat，得到以上的temporal feature

   <div align=center><img src="./resources/skeletons.png" /></div>

#### 2.4 细节阐述
> a)	使用Cha-cha部分数据进行训练
>     
> b)	数据集中包含start/end_position，是由舞蹈人员给出开始/结束时间，通过fps计算。 其中start、end都是比较主观的，使用librosa重新提取节拍，然后让start等于原始start之后最近的一个拍，end等于start+动作的frame length。
>     
> c)	用新的start，end截取music帧，使之与motion帧对齐。

### 3. 复现论文和改进


> 虽然图上画的是Acoustic features作为输入，但是实际上论文中写的是Acoustic features + temporal features 作为输入，经过全连接层增加模型的非线性，然后将编码后的features输入3-layer LSTM, 每个time-step输出ht，再通过全连接层预测动作序列mt。
>
> 文中没有对LSTM前后MLP层的深度、宽度、激活函数作任何描述，我在实现的过程中使用深度为2，宽度64的全连接层，每层的输出用LeakyRelu的方式进行激活。


### 4. 对比实验

Overlap|Index | Model | Strategy | Result |
|:-|:- | :- | :- | :- |
|No|1 | LSTM+AutoEncoder | Base Line(Encoder LSTM, input dim:16, outputdim: 8, hidden_size:30, no dropout, seq_len:20, num_layer=3 Decoder LSTM, input dim: 8, outputdim:16, hidden_size:30, no dropout, seq_len:120, num_layer=3) | Slower convergence than naïve approach, but the result is better.(0.35 to 0.6) |  
||2 | LSTM+AutoEncoder+temporal indexes | Same as 1 | To average |  
||3 | LSTM+AutoEncoder+masking | Same as 1 | Not converge |  
||4 | LSTM+AutoEncoder+masking+temporal indexes | Same as 1 | Not converge |  
|yes|5 | LSTM+AutoEncoder+masking+temporal indexes | Same as 1 | Lot better than 4 |  
||6 | LSTM+AutoEncoder+temporal+masking | Same as 1 | not good as 4 |  
||7 | GRU+temporal+masking | Same as 1 | not good as 4 |  
||8 | GRU+AutoEncoder+temporal+masking | Same as 1 | as good as4 |  

> 因此，根据实验结果，最好的方案是GRU/LSTM + AutoEncoder+masking+temporal indexes + overlap

<div align=center><img src="./resources/Figure_1.png" /></div>
<div align=center>Figure 1: Overlap + per_dim_normalize</div>

<div align=center><img src="./resources/Figure_2.png" /></div>
<div align=center>Figure 2: No-Overlap + per_dim_normalize</div>

<div align=center><img src="./resources/Figure_3.png" /></div>
<div align=center>Figure 3: Overlap + per_feature_normalize</div>

### 5. 最终报告

[最终报告pdf](./final.pdf)

[补充说明pdf](./resources/supp.pdf)

[结果视频1](./resources/DANCE_R_1.mp4)

[结果视频2](./resources/DANCE_C_9.mp4)

更多结果可以查看链接：https://pan.baidu.com/s/13BpVkpnUrZIC21kgKiutTg 提取码:ysop
