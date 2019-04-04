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
  3.复现论文 | Apr.04 | T
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
### 创建数据集
```
  def load_dataset_small(num_images_per_class_train=10, num_images_test=500):
      #Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.

      X_train = []
      X_test = []
    
      # Create training set.
      for c in os.listdir(TRAIN_DIR):
          c_dir = os.path.join(TRAIN_DIR, c, 'images')
          c_imgs = os.listdir(c_dir)
          random.shuffle(c_imgs)
          for img_name_i in c_imgs[0:num_images_per_class_train]:
              img_i = image.load_img(os.path.join(c_dir, img_name_i))
              x = image.img_to_array(img_i)
              X_train.append(x)
      random.shuffle(X_train)
    
      # Create test set.
      test_dir = os.path.join(TEST_DIR, 'images')
      test_imgs = os.listdir(test_dir)
      random.shuffle(test_imgs)
      for img_name_i in test_imgs[0:num_images_test]:
          img_i = image.load_img(os.path.join(test_dir, img_name_i))
          x = image.img_to_array(img_i)
          X_test.append(x)

      # Return train and test data as numpy arrays.
      return np.array(X_train), np.array(X_test)
```
### 构建模型
```
def make_model(input_size):
    input_S = Input(shape=(input_size))
    input_C= Input(shape=(input_size))
    
    encoder = make_encoder(input_size)
    
    decoder = make_decoder(input_size)
    decoder.compile(optimizer='adam', loss=rev_loss)
    decoder.trainable = False
    
    output_Cprime = encoder([input_S, input_C])
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer='adam', loss=full_loss)
    
    return encoder, decoder, autoencoder
```
### 训练
```
NB_EPOCHS = 1000
BATCH_SIZE = 32

m = input_S.shape[0]
loss_history = []
for epoch in range(NB_EPOCHS):
    np.random.shuffle(input_S)
    np.random.shuffle(input_C)
    
    t = tqdm(range(0, input_S.shape[0], BATCH_SIZE),mininterval=0)
    ae_loss = []
    rev_loss = []
    for idx in t:
        
        batch_S = input_S[idx:min(idx + BATCH_SIZE, m)]
        batch_C = input_C[idx:min(idx + BATCH_SIZE, m)]
        
        C_prime = encoder_model.predict([batch_S, batch_C])
        
        ae_loss.append(autoencoder_model.train_on_batch(x=[batch_S, batch_C],
                                                   y=np.concatenate((batch_S, batch_C),axis=3)))
        rev_loss.append(reveal_model.train_on_batch(x=C_prime,
                                              y=batch_S))
        
        # Update learning rate
        K.set_value(autoencoder_model.optimizer.lr, lr_schedule(epoch))
        K.set_value(reveal_model.optimizer.lr, lr_schedule(epoch))
        
        t.set_description('Epoch {} | Batch: {:3} of {}. Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(epoch + 1, idx, m, np.mean(ae_loss), np.mean(rev_loss)))
    loss_history.append(np.mean(ae_loss))
```
![train_process](https://github.com/jialei0701/ANN/blob/master/%E5%BA%B7%E8%87%B3%E7%85%8A21821222/result/train_process.JPG)
### 保存模型
`autoencoder_model.save_weights('model.hdf5')`
### 完整代码
[hiding_image](https://github.com/jialei0701/ANN/blob/master/%E5%BA%B7%E8%87%B3%E7%85%8A21821222/hiding_image.py)
