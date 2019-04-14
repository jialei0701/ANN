# -*- coding: utf-8 -*-
import tensorflow as tf
import argparse
from triplet_loss import batch_all_triplet_loss
from triplet_loss import batch_hard_triplet_loss
import load_data as ld
import json
import inception_resnet_v1 as net
import numpy as np
import calvec as cal
import EER as eer
'''参数， 指定数据地址，和保存模型地址'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', type=str, help="数据地址")
parser.add_argument(
    '--model_dir', default='experiment/model', type=str, help="模型地址")
parser.add_argument(
    '--model_config', default='experiment/params.json', type=str, help="模型参数")


def train_input_fn(data_dir, params):
    """Train input function for the dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (dict) contains hyperparameters of the model
    """
    dataset = ld.train(data_dir)
    dataset = dataset.shuffle(5000)  
    dataset = dataset.repeat(
        params['num_epochs'])  # repeat for multiple epochs
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.prefetch(
        1)  # make sure you always have one batch ready to serve

    train_iterator = dataset.make_one_shot_iterator()
    dataset = train_iterator.get_next()
    return dataset


def test_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (dict) contains hyperparameters of the model
    """
    dataset = ld.test(data_dir)
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.prefetch(
        1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn2(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (dict) contains hyperparameters of the model
    """
    dataset = ld.test2(data_dir)
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.prefetch(
        1)  # make sure you always have one batch ready to serve
    return dataset


def build_model(is_training, images, params):
    '''
       建立模型
       ----------------------------
       Args：
          is_training: bool, 是否是训练阶段，可以从mode中判断
          images：     (batch_size, 120*250*1), 输入数据
          params:      dict, 一些超参数

       Returns:
          out: 输出的embeddings, shape = (batch_size, 128)
    '''
    out = net.inference(images, 0.8, phase_train=is_training)
    out = tf.nn.l2_normalize(out, dim=1)

    return out


def my_model(features, labels, mode, params):
    '''
       model_fn指定函数，构建模型，训练等
       ---------------------------------
       Args:
          features: 输入，shape = (batch_size, 30000)
          labels:   输出，shape = (batch_size, )
          mode:     str, 阶段
          params:   dict, 超参数
    '''
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    images = features
    images = tf.reshape(
        images, shape=[-1, params['image_size1'], params['image_size2'],
                       1])  # reshape (batch_size, 250, 120, 1)
    with tf.variable_scope("model"):
        embeddings = build_model(is_training, images, params)  # 简历模型

    # -------------------------------------------predict---------------------------------------

    if mode == tf.estimator.ModeKeys.PREDICT:  # 如果是预测阶段，直接返回得到embeddings
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # -----------------------------------------------------------------------------------------
    '''调用对应的triplet loss'''
    labels = tf.cast(labels, tf.int64)
    if params['triplet_strategy'] == 'batch_all':
        loss, fraction = batch_all_triplet_loss(
            labels,
            embeddings,
            margin=params['margin'],
            squared=params['squared'])
    elif params['triplet_strategy'] == 'batch_hard':
        loss = batch_hard_triplet_loss(
            labels,
            embeddings,
            margin=params['margin'],
            squared=params['squared'])
    else:
        raise ValueError("triplet_strategy 配置不正确: {}".format(
            params['triplet_strategy']))

    embedding_mean_norm = tf.reduce_mean(tf.norm(
        embeddings, axis=1))  # 这里计算了embeddings的二范数的均值

    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)
    with tf.variable_scope("metrics"):
        eval_metric_ops = {
            'embedding_mean_norm': tf.metrics.mean(embedding_mean_norm)
        }
        if params['triplet_strategy'] == 'batch_all':
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(
                fraction)

    # r_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # loss = loss + r_loss

    # ---------------------------------------eval--------------------------------------

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # ---------------------------------------------------------------------------------

    tf.summary.scalar('loss', loss)
    if params['triplet_strategy'] == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)
    tf.summary.image('train_image', images, max_outputs=1)  # 1代表1个channel

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    global_step = tf.train.get_global_step()

    if params['use_batch_norm']:
        '''如果使用BN，需要估计batch上的均值和方差，tf.get_collection(tf.GraphKeys.UPDATE_OPS)就可以得到
        tf.control_dependencies计算完之后再进行里面的操作
        '''
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):

    flag = 0
    res_eer = 100
    th = 0
    args = parser.parse_args(argv[1:])
    tf.logging.info("创建模型....")
    with open(args.model_config) as f:
        params = json.load(f)

    config = tf.estimator.RunConfig(
        model_dir=args.model_dir, tf_random_seed=100, save_checkpoints_steps=500, keep_checkpoint_max=3)  # config

    cls = tf.estimator.Estimator(
        model_fn=my_model, config=config, params=params)  # 建立模型

    # ----------------------------------trian ---------------------------------
    if flag == 1:
        for i in range(30):
            tf.logging.info("-------------------there are {} epochs....".format(
                params['num_epochs']))

            cls.train(
                input_fn=lambda: train_input_fn(args.data_dir, params))  # 训练模型，指定输入
            

            tf.logging.info("predice........")
            pred = cls.predict(
                input_fn=lambda: test_input_fn(args.data_dir, params))  # 测试模型，指定输入
            
            vec = np.empty((7191, 128), dtype=float)
            cnt = 0
            for p in pred:
                temp = p['embeddings']
                vec[cnt] = temp
                cnt += 1
                # if cnt > 99:
                #     break
            
            np.savetxt("embeddings/model.txt", vec)

            pred = cls.predict(
                input_fn=lambda: test_input_fn2(args.data_dir, params))  # 测试模型，指定输入

            vec = np.empty((11241, 128), dtype=float)
            cnt = 0
            for p in pred:
                temp = p['embeddings']
                vec[cnt] = temp
                cnt += 1
            
            np.savetxt("embeddings/vec.txt", vec)

            cal.fun1()
            cur_eer, cur_th = eer.find_eer()
            if cur_eer <= res_eer and cur_eer != 0:
                res_eer = cur_eer
                th = cur_th

            print("-----------------------------------------")
            print("cur_eer: ", cur_eer)
            print("cur_th: ", cur_th)
            print("res_eer: ", res_eer)
            print("th: ", th)
            print("-----------------------------------------")

    # --------------------------------- eval ----------------------------------
    # tf.logging.info("evaluate model....")
    # res = cls.evaluate(
    #     input_fn=lambda: test_input_fn(args.data_dir, params))  # 测试模型，指定输入
    # for key in res:
    #     print("评价---{} : {}".format(key, res[key]))

    # # --------------------------------- predict -------------------------------
    # flag = 0
    if flag == 0:
        tf.logging.info("predice........")
        pred = cls.predict(
            input_fn=lambda: test_input_fn(args.data_dir, params))  # 测试模型，指定输入
        
        vec = np.empty((15996, 128), dtype=float)
        cnt = 0
        for p in pred:
            temp = p['embeddings']
            vec[cnt] = temp
            cnt += 1
            # if cnt > 99:
            #     break
        
        np.savetxt("embeddings/model_200.txt", vec)

        # pred = cls.predict(
        # input_fn=lambda: test_input_fn2(args.data_dir, params))  # 测试模型，指定输入

        # vec = np.empty((11241, 128), dtype=float)
        # cnt = 0
        # for p in pred:
        #     temp = p['embeddings']
        #     vec[cnt] = temp
        #     cnt += 1
        
        # np.savetxt("embeddings/vec.txt", vec)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
