"""
Original: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
"""
from __future__ import print_function

import argparse
import json
import os

import keras
from keras.callbacks import LambdaCallback
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D

from parse_layer_spec import add_layers
from utils import use_valohai_inputs


def train(cli_params):
    batch_size = cli_params.batch_size
    num_classes = cli_params.num_classes
    epochs = cli_params.epochs
    data_augmentation = cli_params.data_augmentation
    model_layers = cli_params.model_layers

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

    # Define layers from Valohai parameters for more complex hyperparameter tuning.
    model = add_layers(model, model_layers)

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Initiate RMSprop optimizer.
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop.
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
        )
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            fill_mode='nearest',  # set mode for filling points outside the input boundaries
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            rescale=None,  # set rescaling factor (applied before any other transformation)
            preprocessing_function=None,  # set function that will be applied on each input
            data_format=None,  # image data format, either "channels_first" or "channels_last"
            validation_split=0.0,  # fraction of images reserved for validation (strictly between 0 and 1)
        )

        # Compute quantities required for feature-wise normalization.
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

        # We use custom JSON logging to integrate with Valohai metadata system.
        json_logging_callback = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: print('Epoch %s/%s' % ((int(epoch) + 1), epochs)),
            on_epoch_end=lambda epoch, logs: print(json.dumps({
                'epoch': int(epoch) + 1,
                'loss': str(logs['loss']),
                'accuracy': str(logs['accuracy']),
                'val_loss': str(logs['val_loss']),
                'val_accuracy': str(logs['val_accuracy']),
            })),
            # Add occasional batch logging so we can quickly see it is progressing.
            on_batch_begin=lambda batch, logs: (print('Batch %s' % batch) if batch % 100 == 0 else None),
        )

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4,
            callbacks=[json_logging_callback],
            verbose=0,  # disable default logging, it is unnecessarily noisy
        )

    # Save model and weights.
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './')
    output_file = os.path.realpath(os.path.join(outputs_dir, 'my_model.h5'))
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)

    print('Saving trained model to %s' % output_file)
    model.save(output_file)

    # Score trained model.
    print('Scoring the trained model.')
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(json.dumps({'test_loss': scores[0], 'test_accuracy': scores[1]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_augmentation', type=int, required=True)
    parser.add_argument('--model_layers', type=str, required=True)
    cli_parameters, unparsed = parser.parse_known_args()
    use_valohai_inputs(
        valohai_input_name='cifar-10-batches-py',
        input_file_pattern='*.tar.gz',
        keras_cache_dir='datasets',
        keras_example_file='cifar-10-batches-py.tar.gz',
    )
    train(cli_parameters)
