import tensorflow as tf
import time
import os
import model
import reader
import losses
import sys

batch_size = 4
image_size = 256
style_layers = ["conv1_2",
                "conv2_2",
                "conv3_3",
                "conv4_3",]
content_layers =  ["conv3_3",]
vgg16_ckpt_path = 'vgg_16.ckpt'
style_weight = 200.0
content_weight = 1.0
tv_weight = 0.0



def get_style_feature():
    with tf.Graph().as_default():
        img_bytes = tf.read_file(style_path)
        if style_path.lower().endswith('png'):
            style_image = tf.image.decode_png(img_bytes)
        else:
            style_image = tf.image.decode_jpeg(img_bytes)
        style_image = reader.prepose_image(style_image,image_size,image_size)
        style_image = tf.expand_dims(style_image, 0)
        style_net = model.load_model(style_image,vgg16_ckpt_path)
        features = []
        for layer in style_layers:
            feature = style_net[layer]
            feature = tf.squeeze(losses.gram(feature), [0])  # remove the batch dimension

            features.append(feature)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
            # Make sure the 'generated' directory is exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            # Indicate cropped style image path
            save_file = 'generated/target_style_' + style_path

            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = reader.mean_add(style_image[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            return sess.run(features)

def main():
    # Make sure the training path exists.
    training_path = 'models/log/'
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)
    style_features = get_style_feature()



    with tf.Graph().as_default():
        train_image = reader.get_train_image(batch_size,image_size,image_size,dataset_path)
        generated = model.net(train_image)
        processed_generated = [reader.prepose_image(image,image_size,image_size)
                               for image in tf.unstack(generated, axis=0, num=batch_size)
                               ]
        processed_generated = tf.stack(processed_generated)
        net = model.load_model(tf.concat([processed_generated, train_image], 0),vgg16_ckpt_path)
        with tf.Session() as sess:
            """Build Losses"""
            content_loss = losses.content_loss(net, content_layers)
            style_loss, style_loss_summary = losses.style_loss(net, style_features, style_layers)
            tv_loss = losses.total_variation_loss(generated)  # use the unprocessed image

            loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tv_loss

            # Add Summary for visualization in tensorboard.
            """Add Summary"""
            tf.summary.scalar('losses/content_loss', content_loss)
            tf.summary.scalar('losses/style_loss', style_loss)
            tf.summary.scalar('losses/regularizer_loss', tv_loss)

            tf.summary.scalar('weighted_losses/weighted_content_loss', content_loss * content_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', style_loss * style_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * tv_weight)
            tf.summary.scalar('total_loss', loss)

            for layer in style_layers:
                tf.summary.scalar('style_losses/' + layer, style_loss_summary[layer])
            tf.summary.image('generated', generated)
            tf.summary.image('origin', tf.stack([reader.mean_add(image)
                                                 for image in tf.unstack(train_image, axis=0, num=batch_size)
                                                ]))
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(training_path)

            """Prepare to Train"""
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=tf.trainable_variables())

            saver = tf.train.Saver(tf.trainable_variables())

            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

            # Restore variables for training model if the checkpoint file exists.
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    loss_c, loss_s = sess.run([content_loss,style_loss])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    """logging"""
                    if step % 10 == 0:
                        print('step: %d, content Loss %f, style Loss %f, total Loss %f, secs/step: %f'
                              % (step, loss_c, loss_s, loss_t, elapsed_time))
                    """summary"""
                    if step % 25 == 0:
                        print('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt'), global_step=step)
            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, 'fast-style-model.ckpt-done'))
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("please input style image path and train datasets!")
        exit()
    style_path = sys.argv[1]
    dataset_path = sys.argv[2]
    main()
# `python train.py /path/to/style.jpg /path/to/train/`