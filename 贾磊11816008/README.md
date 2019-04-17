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
3.复现论文|Mar. 28|√
4.完成对比实验|Apr. 4|√
5.形成报告|Apr. 11|√

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

### 数据集

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/datasets.jpg)

### Framework

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/framework.jpg)

### Convolution layer

&emsp; 输入residue-wise数据S左右填补（m-1）的unuseful residue，转换为矩阵M（类图像像素数据）；

&emsp; 输出为矩阵X，其中X<sub>i,k</sub>表示第k个motif detector在第i个位置的得分；

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/conv_layer.jpg)

### Rectification layer

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/rectification_layer.jpg)

&emsp; 过滤非高效motif特征

### Pooling layer

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/pooling_layer.jpg)

&emsp;最大池化

### Neural network layer

&emsp; 综合motif特征、sequence特征、evolutionary特征进行预测。
&emsp; 采用dropout technique避免overfitting。


### 不同特征比较

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/ROC.jpg)

### 方法间比较

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/compare.jpg)

&emsp; Sensitivity (SN), Specificity (SP), Strength (ST), Accuracy (ACC), and Mathews Correlation Coefficient (MCC).

### Motif特征有效性
Explanation for the effectiveness of motif features for the prediction of DNA-binding residue

Discriminant power (DP) of a motif t in CNNsite is calculated by the following formula:
![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/DP.jpg)

We find that the residues R, K, G are the important compositions of these motifs. This finding is consistent with the study of Szilgyi and Skolnick, in which they found that R, A, G, K and D are important for the formation of protein-DNA interactions. The importance of R for the formation of protein-DNA interactions is further confirmed by Sieber and Allemanns work, which states that R can indirectly interact with DNA by interacting with both the phosphate backbone and the carboxylate of E(345). Since these residues are important for the formation of protein-DNA interactions, we speculate that they often occur in the context of the DNA-binding residues and their occurrences are important features for prediction.

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/TOP15.jpg)

![avator](https://github.com/jialei0701/ANN/blob/master/%E8%B4%BE%E7%A3%8A11816008/proposition.jpg)

---
## 3. 复现论文
见代码

```python
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def train_conv_net(datasets,
		   pssms,
		   physichemics,
                   U,
                   img_w=20, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,150,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-2  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    z = T.ivector('z')
    U = np.asarray(U,dtype=theano.config.floatX)
    pssms=np.asarray(pssms,dtype=theano.config.floatX)
    physichemics=np.asarray(physichemics,dtype=theano.config.floatX)
    Words = theano.shared(value = U, name = "Words")
    PSSMs = theano.shared(value = pssms,name = "PSSMs")
    PHYSICHEMICs = theano.shared(value = physichemics,name = "PHYSICHEMICs")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w,dtype = np.float32)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    pssm_input = PSSMs[T.cast(z.flatten(),dtype="int32")].reshape((z.shape[0],1,1,PSSMs.shape[1]))
    physichemic_input = PHYSICHEMICs[T.cast(z.flatten(),dtype="int32")].reshape((z.shape[0],1,1,PHYSICHEMICs.shape[1]))
    layer1_inputs.append(pssm_input.flatten(2))
    layer1_inputs.append(physichemic_input.flatten(2))
    layer1_input=T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)+220+136
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = np.asarray(datasets[1][:,:img_h],dtype=np.float32) 
    test_set_y = np.asarray(datasets[1][:,img_h],"int32")
    test_set_z = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y, train_set_z = shared_dataset((train_set[:,:img_h],train_set[:,img_h],train_set[:,-1]))
    val_set_x, val_set_y, val_set_z = shared_dataset((val_set[:,:img_h],val_set[:,img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size],
	    z: val_set_z[index * batch_size: (index + 1) * batch_size]
								     })
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
		z: train_set_z[index * batch_size: (index + 1) * batch_size]
									    })               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size],
	    z: train_set_z[index*batch_size:(index+1)*batch_size]
								})     
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    pssm_input = PSSMs[T.cast(z.flatten(),dtype="int32")].reshape((z.shape[0],1,1,PSSMs.shape[1]))
    physichemic_input = PHYSICHEMICs[T.cast(z.flatten(),dtype="int32")].reshape((z.shape[0],1,1,PHYSICHEMICs.shape[1]))
    test_pred_layers.append(pssm_input.flatten(2))
    test_pred_layers.append(physichemic_input.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict_p(test_layer1_input)
    test_error=error(y,test_y_pred)
    #test_error = T.mean(T.neq(test_y_pred, y))
    y_prob=get_y_prob(test_y_pred,y)
    test_model_all = theano.function([x,y,z],y_prob)
    #test_model_all = theano.function([x,y,z],test_error)   
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch %i, train perf %f %%, val perf %f,' % (epoch, train_perf * 100., val_perf*100.,))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_set_y,test_set_z)     
            test_perf = test_loss          
    return test_perf,best_val_perf

def get_y_prob(test_y_pred,y):
    return test_y_pred,y

def error(y,test_y_pred):
    """
    calculate the testing performance
    """
    p=T.cast(T.sum(y),dtype="float32")
    n=T.cast(T.sum(T.ones_like(y))-T.sum(y),dtype="float32")
    fn=T.cast(T.sum(T.lt(test_y_pred,y)),dtype="float32")
    fp=T.cast(T.sum(T.lt(y,test_y_pred)),dtype="float32")
    tp=p-fn
    tn=n-fp
    acc=(tp+tn)/(p+n)
    sn=tp/p
    sp=tn/n
    st=(sn+sp)/2
    d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=(tp*tn-fp*fn)/d
    return acc,sn,sp,st,mcc

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y, data_z = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
	shared_z = theano.shared(np.asarray(data_z,
                                                dtype=theano.config.floatX),
                                  borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_z,'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=20, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=20, filter_h=5, index=0,num_of_files=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    pos=0
    neg=0
    total_neg=0
    train, test = [], []
    sample_index=0
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
	sent.append(sample_index)
        sample_index=sample_index+1
        if rev["split"]==cv:           
            test.append(sent)        
        else:
            if rev["y"]==1:
               train.append(sent)
               pos=pos+1
            elif rev["y"]==0 and (total_neg%num_of_files==index or total_neg%num_of_files==(index+1)%num_of_files):
               train.append(sent)
	       neg=neg+1
               total_neg=total_neg+1
	    else:
	       total_neg=total_neg+1
    print "index: "+str(index)+" train positive number: "+str(pos)+" train negative number: "+str(neg) 
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]
 
def calc_perf(test_y_prob,y,tag):
    y_pred=[]
    for i in range(0,len(test_y_prob)):
        if test_y_prob[i][0]>test_y_prob[i][1]:
	   y_pred.append(0)
	else:
	   y_pred.append(1)
    #write the classification result into file
    filename = "./Result/CNN_PSSM_PHY_PDNA224_ROC.txt"
    if tag == 1:
       output = open(filename,'w+')
       for i in range(0,len(test_y_prob)):
	   line = str(y[i]) + "	" + str(test_y_prob[i][0]) + "\n"
	   output.write(line)
	   output.flush()
       output.close()
    tp=0.0
    tn=0.0
    fn=0.0
    fp=0.0
    for i in range(0,len(y)):
        if y_pred[i]==1 and y[i]==1:
           tp=tp+1
        elif y_pred[i]==0 and y[i]==1:
           fn=fn+1
        elif y_pred[i]==1 and y[i]==0:
           fp=fp+1
        elif y_pred[i]==0 and y[i]==0:
           tn=tn+1
    print "tp: "+str(tp)+" fn: "+str(fn)+"  tn: "+str(tn)+"  fp: "+str(fp)
    acc=(tp+tn)/(tp+fp+tn+fn)
    sn=tp/(tp+fn)
    sp=tn/(tn+fp)
    st=(sn+sp)/2
    d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=(tp*tn-fp*fn)/d
    return acc,sn,sp,st,mcc

   
if __name__=="__main__":
    output = open("Result/CNN_PSSM_PHY_PDNA224.txt","w")
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab, pssms,physichemics = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    print str(len(pssms[0]))+" "+str(len(physichemics[0]))
    #print pssms
    #pssms=np.array(pssms,dtype="float32")
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    accs=[]
    sns=[]
    sps=[]
    sts=[]
    mccs=[]
    file_num=11
    r = range(0,10)    
    for i in r:
	results=[]
	ensemble=[]
	perfs=[]
	val_perfs=[]
        for j in range(0,file_num):
            datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=11,k=20, filter_h=6, index=j,num_of_files=file_num)
	    #print datasets
            result,val_perf = train_conv_net(datasets,
				    pssms,
				    physichemics,
                                    U,
                                    lr_decay=0.95,
                                    filter_hs=[2,3,4,5,6],
                                    conv_non_linear="relu",
                                    hidden_units=[100,150,2], 
                                    shuffle_batch=True, 
                                    n_epochs=50, 
                                    sqr_norm_lim=9,
                                    non_static=non_static,
                                    batch_size=50,
                                    dropout_rate=[0.5])
	    results.append(result)
	    val_perfs.append(val_perf)
            perf=calc_perf(result[0],result[1],0)
	    perfs.append(perf)
            print "cv: " + str(i) + ", index: " +str(j) + ", acc: " + str(perf[0]) + ", sn: " + str(perf[1]) + ", sp: " + str(perf[2]) + ", st: " + str(perf[3]) + ", mcc: " + str(perf[4])
        for k in range(0,len(results[0][0])):
	    prob=[]
	    pos_prob=0
	    neg_prob=0
	    for m in range(0,file_num):
	        neg_prob+=results[m][0][k][0]
	        pos_prob+=results[m][0][k][1]
	    prob.append(neg_prob)
	    prob.append(pos_prob)
	    ensemble.append(prob)
	perf=calc_perf(ensemble,results[0][1],1)
        print "cv: " + str(i) + ", index: ensemble" + ", acc: " + str(perf[0]) + ", sn: " + str(perf[1])+ ", sp: " + str(perf[2]) + ", st: " + str(perf[3]) + ", mcc: " + str(perf[4])
	line = "cv: " + str(i) + ", index: ensemble" + ", acc: " + str(perf[0]) + ", sn: " + str(perf[1])+ ", sp: " + str(perf[2]) + ", st: " + str(perf[3]) + ", mcc: " + str(perf[4])+"\n"
	output.write(line)
	output.flush()    
	accs.append(perf[0])
        sns.append(perf[1])
        sps.append(perf[2])
        sts.append(perf[3])
        mccs.append(perf[4])  
    print "acc:"+str(np.mean(accs))+", sn:"+str(np.mean(sns))+", sp:"+str(np.mean(sps))+", st:"+str(np.mean(sts))+", mcc:"+str(np.mean(mccs))
    line = "acc:"+str(np.mean(accs))+", sn:"+str(np.mean(sns))+", sp:"+str(np.mean(sps))+", st:"+str(np.mean(sts))+", mcc:"+str(np.mean(mccs))+"\n"+"\n"
    output.write(line)
    output.flush()
    output.close()
```

---
## 4. 对比实验


---


