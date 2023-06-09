import os
import boto3
import numpy as np

from keras import backend
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export

s3 = boto3.client('s3')
# s3 = boto3.client(
#     service_name='s3',
#     endpoint_url='https://bucket.vpce-0dc90772aeda024fb-m0bi39ow.s3.us-east-2.vpce.amazonaws.com'
#     )

bucket_name = '775-bucket'
local_folder_path = './datasets/cifar-10-batches-py'

def download_from_s3():
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)

    objects = s3.list_objects(Bucket=bucket_name)['Contents']

    for obj in objects:
        local_file_path = os.path.join(local_folder_path, obj['Key'])        
        s3.download_file(bucket_name, obj['Key'], local_file_path)
        print('file download to path : ', local_file_path)

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
