import tensorflow as tf


def _parse_record(record):
    features = {
        'data': tf.FixedLenFeature([250 * 120], dtype=tf.float32),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    parsed_features = tf.parse_single_example(record, features=features)

    data = parsed_features['data']
    label = parsed_features['label']
    return data, label


def dataset(input_file):
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_record)

    return dataset


def train(directory):
    """tf.data.Dataset object for training data."""
    return dataset('./data/train_600.tfrecord')


def test(directory):
    """tf.data.Dataset object for test data."""
    return dataset('./data/test_200.tfrecord')


def test2(directory):
    """tf.data.Dataset object for test data."""
    return dataset('./data/test_400.tfrecord')


if __name__ == "__main__":
    d = train("data")
