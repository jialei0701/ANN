# 固定风格任意内容图像风格迁移
基于python3 & tensorflow实现的 **固定风格任意内容图像风格迁移** ，是对《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》的实现。



## 训练步骤：
- 1.下载 [vgg16.ckpt](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)，并放在代码同级文件夹下，命名为'vgg_16.ckpt'。
- 2.下载 [COCO数据集](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)，假设解压后的文件夹路径为 '/path/to/train/'。
- 3.准备好一张风格图像，图像大小不限制，最好使用256*256左右的分辨率图像，假设图像路径为 '/path/to/style.jpg'。
- 4.配置好环境之后，运行命令：

    ```bash
    # train.py程序接受两个命令行参数，第一个是风格图像的路径，第二个是训练的数据集的路径。
    python train.py /path/to/style.jpg /path/to/train/
    ```
    
- 5.在训练期间，可以使用tensorboard查看训练过程：

    ```
    tensorboard --logdir=models/log/
    ```
- 6.训练结束后，在model/log目录下，找到'fast-style-model.ckpt-done'开头的四个文件，这就是最终保存的模型。

## 测试步骤：
- 1.假设需要转换的图像路径是'img/test.jpg'，训练模型的保存路径是'model/log/'，那么运行命令：

    ```bash
    # eval.py程序接受两个命令行参数，第一个是测试图像的路径，第二个是模型保存的路径。
    python eval.py img/test.jpg model/log/fast-style-model.ckpt-done
    ```
- 2.生成的结果在'generated/res.jpg'。

