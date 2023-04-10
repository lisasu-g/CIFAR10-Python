import os
import boto3
import numpy as np

from keras import backend
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export

s3 = boto3.resource('s3')
bucket_name = '775-bucket'
local_path = 'datasets/cifar-10-batches-py'



def download_from_s3():
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.all():
        filename = obj.key.split('/')[-1]
        # bucket.download_file(obj.key, filename)
        s3.download_file(bucket_name, obj.key, os.path.join(local_path, filename))
        print(f"Downloaded {obj.key}")

def load_local_data():
    path = "./datasets/cifar-10-batches-py"

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)
