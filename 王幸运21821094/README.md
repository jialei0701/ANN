# 个人信息
姓名：王幸运 <br>
学号：21821094 <br>
邮箱：2968555822@qq.com <br>
电话：15956751830<br>
# 1. 选择论文
**Title:**<br>
Deep Adaptive Sampling for Low Sample Count Rendering <br><br>
**Abstract:**<br>
Recently, deep learning approaches have proven successful at removing noise from Monte Carlo (MC) rendered images at extremely
low sampling rates, e.g., 1-4 samples per pixel (spp). While these methods provide dramatic speedups, they operate on uniformly
sampled MC rendered images. However, the full promise of low sample counts requires both adaptive sampling and reconstruction/denoising
. Unfortunately, the traditional adaptive sampling techniques fail to handle the cases with low sampling rates, since there is
insufficient information to reliably calculate their required features, such as variance and contrast. In this paper, we address
this issue by proposing a deep learning approach for joint adaptive sampling and reconstruction of MC rendered images with extremely
low sample counts. Our system consists of two convolutional neural networks (CNN), responsible for estimating the sampling map and 
denoising, separated by a renderer. Specifically, we first render a scene with one spp and then use the first CNN to estimate a sampling
map, which is used to distribute three additional samples per pixel on average adaptively. We then filter the resulting render with the
second CNN to produce the final denoised image. We train both networks by minimizing the error between the denoised and ground truth 
images on a set of training scenes. To use backpropagation for training both networks, we propose an approach to effectively compute 
the gradient of the renderer. We demonstrate that our approach produces better results compared to other sampling techniques. On average,
our 4 spp renders are comparable to 6 spp from uniform sampling with deep learning-based denoising. Therefore, 50% more uniformly 
distributed samples are required to achieve equal quality without adaptive sampling.<br><br>
**摘要:**<br>
深度学习方法可以在以极低采样率（例如，每像素1-4个样本（spp））的蒙特卡洛（MC）渲染图像中去除噪声。
这些方法提供了明显的加速，但它们只对均匀采样的MC渲染图像进行操作。然而，低样本数的图像需要自适应采样和去噪。
令人遗憾的是，传统的自适应采样技术无法处理低采样率的情况，因为没有足够的信息来计算其所需的特征，例如方差和对比度。在本
文中，我们通过提出一种深度学习方法来解决这个问题，该方法用于自适应采样和重建具有极低样本数的MC渲染图像。我们的系统
由两个卷积神经网络（CNN）组成，负责估计采样图和去噪，他们由渲染器分隔。具体来说，我们首先使用一个spp渲染场景，然后
使用第一个CNN来估计采样图，该图用于平均自适应地分配每个像素的三个附加样本。然后，我们用第二个CNN过滤得到的渲染图，
以产生最终的去噪图像。我们通过最小化一组训练场景中的去噪和真实图像之间的误差来训练两个网络。为了使用反向传播来训练两个
网络，我们提出了一种有效计算渲染器梯度的方法。我们证明，与其他采样技术相比，我们的方法产生了更好的结果，我们的4 spp渲染
与基于深度学习去噪的均匀采样的6 spp质量相当。因此，在没有自适应采样的情况下，额外需要50％均匀分布的样本以实现相同的质量。
<br><br>

# 2. 精读论文，理解模型<br>
  共有两个CNN,第一个CNN(Sampling Map Estimator)用于估算采样图，他的输入是1spp的图像，还有一些其他的辅助缓冲信息，输出是一个采样图；第二个CNN（Denoiser）用于给图像去燥，输入是4spp的图像，输出是最终的图像。<br>
  ## 工作过程图
  ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/process.png)
  
  ## 网络结构
  ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/cnn.png)
 <br> 这两个CNN都用了 encoder-decoder的结构，该网络共包含5个encoder和decoder单元，中间用bottleneck隔开。每个encoder由两个卷积层和1个max pooing层组成;每个decoder有一个 upsampling层和两个卷积层组成。卷积层的大小都是3X3.在所有的卷积层后面（除了最后一个）都用了ReLU激活函数。
 
 ## 训练这两个网络
 通过以端到端的方式，最小化去燥图像与真实图像之间的误差来训练这两个网络。
 ### 梯度
  ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/gradient1%20.png)
  <br>h是额外的样本点，用于数值微分。Is+h是由Is和Ih所获得的。
  <br>但是这个梯度有很大噪音，所以下面通过大量的h spp的图像来计算梯度。
  ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/gradient2.png)
  <br>然后把右边的代入进去，可以得到:<br>
   ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/gradient3.png)
   <br>然后把累加符号代入到分子上，又可以得到：<br>
   ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/gradient4.png)
   <br>当N->∞，h=0,I∞表示地面真实图像，最终得到：<br>
    ![avator](https://github.com/jialei0701/ANN/blob/master/%E7%8E%8B%E5%B9%B8%E8%BF%9021821094/gradient6.png)
   
