## Personal Information
* **Name**: 辛越 
* **Student ID**: 21821256
* **E-Mail:**: x1046106323@zju.edu.cn

## Schedule
| Task | Due | Done |
| :-- | :-: | :-: |
| 1. 选择论文 | Mar. 14 |  |
| 2. 精读论文，理解模型 | Mar. 21 |  |
| 3. 复现论文 | Apr. 4 |  |
| 4. 完成对比实验 | Apr. 11 |  |
| 5. 形成最后报告 | Apr. 18 |  | 

### 选择论文
[SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud](SqueezeSeg.pdf)<br>
**Author(s)**: Bichen Wu ; Alvin Wan ; Xiangyu Yue ; Kurt Keutzer<br>
**Published in**: [2018 IEEE International Conference on Robotics and Automation (ICRA)](https://ieeexplore.ieee.org/xpl/mostRecentIssue.jsp?punumber=8449910)<br>

* #### Abstract
>We address semantic segmentation of road-objects from 3D LiDAR point clouds. In particular, we wish to detect and categorize instances of interest, such as cars, pedestrians and cyclists. We formulate this problem as a point-wise classification problem, and propose an end-to-end pipeline called SqueezeSeg based on convolutional neural networks (CNN): the CNN takes a transformed LiDAR point cloud as input and directly outputs a point-wise label map, which is then refined by a conditional random field (CRF) implemented as a recurrent layer. Instance-level labels are then obtained by conventional clustering algorithms. Our CNN model is trained on LiDAR point clouds from the KITTI dataset, and our point-wise segmentation labels are derived from 3D bounding boxes from KITTI. To obtain extra training data, we built a LiDAR simulator into Grand Theft Auto V (GTA-V), a popular video game, to synthesize large amounts of realistic training data. Our experiments show that SqueezeSeg achieves high accuracy with astonishingly fast and stable runtime (8.7 ± 0.5 ms per frame), highly desirable for autonomous driving. Furthermore, additionally training on synthesized data boosts validation accuracy on real-world data. Our source code is open-source released. The paper is accompanied by a video containing a high level introduction and demonstrations of this work.

* #### 摘要
>我们解决了3D LiDAR点云对道路上物体的语义分割问题。特别地，我们希望检测和分类感兴趣的实例，例如汽车、行人和骑自行车的人。我们将此问题描述为逐点分类问题，并提出一种基于卷积神经网络（CNN）的端到端流水线SqueezeSeg：CNN将转换后的点云作为输入并直接输出逐点的标签映射，然后被作为重复层实现的条件随机场（CRF）重新定义，之后通过传统的聚类算法获得实例级标签。我们的CNN模型使用KITTI数据集的LiDAR点云训练，我们的逐点分割标签来自于KITTI的3D bounding box。为了获取额外的训练数据，我们在流行的视频游戏“侠盗猎车手V”（GTA-V）构建了一个LiDAR模拟器，以合成大量逼真的实验数据。我们的实验表明，SqueezeSeg达到了很高的精度，具有惊人的速度和稳定的运行时间（8.7 ± 0.5 ms每帧）非常适合自动驾驶。此外，对合成数据的额外训练提高了使用真实世界数据验证的准确性。我们的源代码是开源发布的。本文附有一个视频，其中包括对这项工作的高级内容和展示。
### 精读论文

