# 生物智能算法 神经网络组

## Personal information
- Name:康至煊
- StudentID：21821222
- Email：kzx1995@126.com

## Timeline
  Task | Date | Done
  ------- | ------- | :-------:
  1.选择论文 | Mar.14 | T
  2.精读论文 | Mar.21 | T
  3.复现论文 | Apr.04 |
  4.完成对比试验 | Apr.11 | 
  5.形成报告 | Apr.18 |
*****
## 1.选择论文
#### Title：Hiding Images in Plain Sight:Deep Steganography
> 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA
#### Abstract
> Steganography is the practice of concealing a secret message within another,ordinary, message. Commonly, steganography is used to unobtrusively hide a smallmessage within the noisy regions of a larger image. In this study, we attemptto place a full size color image within another image of the same size. Deepneural networks are simultaneously trained to create the hiding and revealing processes and are designed to specifically work as a pair. The system is trained on images drawn randomly from the ImageNet database, and works well on naturalimages from a wide variety of sources. Beyond demonstrating the successfulapplication of deep learning to hiding images, we carefully examine how the resultis achieved and explore extensions. Unlike many popular steganographic methodsthat encode the secret message within the least significant bits of the carrier image,our approach compresses and distributes the secret image’s representation across all of the available bits.
#### 摘要
> 隐写术是一种在普通的信息中隐藏秘密信息的方法。通常，隐写术被用来在较大图像的噪声区域内隐藏一个小消息。在本研究中，我们尝试将一个全尺寸的彩色图像放置在另一个大小相同的图像中。深层神经网络同时被训练来创建隐藏和揭示过程，并被专门设计成一对。该系统对随机从ImageNet数据库中提取的图像进行培训，并能很好地处理来自各种来源的自然图像。除了演示如何成功地将深度学习应用于隐藏图像之外，我们还仔细研究了结果是如何实现的，并探索了扩展。与许多常用的隐写方法不同，常用方法将秘密消息编码在载波图像中最不重要的比特中，而我们的方法将需要加密的图像压缩和分发在所有可获得的bit中。
*****
## 2.精读论文
#### 数据集: ImageNet
#### 模型架构
![model Architecture](https://github.com/jialei0701/ANN/blob/master/%E5%BA%B7%E8%87%B3%E7%85%8A21821222/%E8%AE%BA%E6%96%87%E5%9B%BE%E7%89%871.JPG)
- preparation network : 当需隐藏图像小于载体图像时，加大需隐藏图像，从而将bit分布在NxN（载体图像大小）对图像进行编码
- main network : 以载体图像和需隐藏图像做为输入，输出隐藏后的容器图像（container image）
- reveal network : 接收container image，移除载体图像，揭示隐藏图像
#### 网络细节
- main network和reveal network为5层卷积层，每层65 filters (50 3x3 filters, 10 4x4 filters and 5 5x5 filters)
- preparation network为2层相同结构的卷积层
#### 结果
![result](https://github.com/jialei0701/ANN/blob/master/%E5%BA%B7%E8%87%B3%E7%85%8A21821222/%E8%AE%BA%E6%96%87%E5%9B%BE%E7%89%872.JPG)

## 3.复现论文
[dataset](https://tiny-imagenet.herokuapp.com/)
