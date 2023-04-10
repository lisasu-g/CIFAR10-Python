import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from loadFromS3 import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('myapp.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def trainImages():
    logger.info('Training logic function starts')

    # use downloaded dataset from FTP server and load data
    # (train_images, train_labels), (test_images, test_labels) =  tf.keras.datasets.cifar10.load_data()

    # download datas from S3
    download_from_s3()
    # use locally loaded data
    (train_images, train_labels), (test_images, test_labels) =  load_local_data()

    logger.info('The download logic is finished.')

    train_images, test_images = train_images / 255.0, test_images / 255.0


    # verify data
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                'dog', 'frog', 'horse', 'ship', 'truck']

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i])
    #     # The CIFAR labels happen to be arrays, 
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()


    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # model.summary()

    # add dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))


    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    logger.info('test_acc: ' + test_acc)
    logger.info('The training logic is finished.')


trainImages()