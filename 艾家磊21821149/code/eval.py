# coding: utf-8
import tensorflow as tf
import reader
import model
import time
import os
import sys

def main(image_file,model_file):

    # Get image's height and width.
    height = 0
    width = 0
    with open(image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.

            image = reader.get_image(image_file, height, width)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            model_file = os.path.abspath(model_file)
            saver.restore(sess, model_file)

            # Make sure 'generated' directory exists.
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    if len(sys.argv)<3:
        print("please input content image path and model path!")
        exit()
    main(sys.argv[1],sys.argv[2])
# python eval.py img/test.jpg model/fast-style-model.ckpt-done
