# -*- coding: utf-8 -*-
"""
@author: Chase
"""

import numpy as np
np.random.seed(31337)
import os
import glob
import cv2
import datetime
import time
from sklearn.cross_validation import KFold
from keras.models import Model
from keras.layers import Input, SpatialDropout2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import log_loss


def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
    return resized


def load_train(img_rows, img_cols):
    X_train = []
    X_train_id = []
    mask_train = []
    start_time = time.time()

    print('Read train images')
    files = glob.glob("./data/train/*[0-9].tif")
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_train.append(img)
        X_train_id.append(flbase[:-4])
        mask_path = "./data/train/" + flbase[:-4] + "_mask.tif"
        mask = get_im_cv2(mask_path, img_rows, img_cols)
        mask_train.append(mask)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, mask_train, X_train_id


def load_test(img_rows, img_cols):
    print('Read test images')
    files = glob.glob("./data/test/*[0-9].tif")
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase[:-4])
        total += 1

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def rle_encode(img, order='F'):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r

    z = ''
    for rr in runs:
        z += str(rr[0]) + ' ' + str(rr[1]) + ' '
    return z[:-1]


def find_best_mask():
    files = glob.glob(os.path.join(".", "data", "train", "*_mask.tif"))
    overall_mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    overall_mask.fill(0)
    overall_mask = overall_mask.astype(np.float32)

    for fl in files:
        mask = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        overall_mask += mask
    overall_mask /= 255
    max_value = overall_mask.max()
    koeff = 0.5
    overall_mask[overall_mask < koeff * max_value] = 0
    overall_mask[overall_mask >= koeff * max_value] = 255
    overall_mask = overall_mask.astype(np.uint8)
    return overall_mask


def create_submission(predictions, test_id, info):
    sub_file = os.path.join('submission_' + info + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    mask = find_best_mask()
    encode = rle_encode(mask)
    subm.write("img,pixels\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > 0.5:
            subm.write(encode)
        subm.write('\n')
    subm.close()


def get_empty_mask_state(mask):
    out = []
    for i in range(len(mask)):
        if mask[i].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)


def read_and_normalize_train_data(img_rows, img_cols):
    train_data, train_target, train_id = load_train(img_rows, img_cols)
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    # Convert to 0 or 1
    train_target = get_empty_mask_state(train_target)
    train_target = np_utils.to_categorical(train_target, 2)
    train_data = train_data.astype('float32')
    train_data = (train_data - 127.5) / 127.5
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data(img_rows, img_cols):
    test_data, test_id = load_test(img_rows, img_cols)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data = (test_data - 127.5) / 127.5
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model(img_rows, img_cols):
    input = Input((1, img_rows, img_cols))
    
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(input)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    pool2 = AveragePooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    
    comb1 = merge([conv2, UpSampling2D(size=(2,2))(conv3)], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(comb1)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    
    comb2 = merge([conv1, UpSampling2D(size=(2,2))(conv4)], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(comb2)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    
    output = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=input, output=output)
    model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy')
    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def getPredScorePercent(train_target, train_id, predictions_valid):
    perc = 0
    for i in range(len(train_target)):
        pred = 1
        if predictions_valid[i][0] > 0.5:
            pred = 0
        real = 1
        if train_target[i][0] > 0.5:
            real = 0
        if real == pred:
            perc += 1
    perc /= len(train_target)
    return perc


def run_cross_validation(nfolds=10):
    img_rows, img_cols = 256, 256
    batch_size = 32
    nb_epoch = 200
    random_state = 42

    train_data, train_target, train_id = read_and_normalize_train_data(img_rows, img_cols)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_index, test_index in kf:
        model = create_model(img_rows, img_cols)
        X_train, X_valid = train_data[train_index], train_data[test_index]
        Y_train, Y_valid = train_target[train_index], train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))


        print('Building data augmentation object...')
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.10,
            height_shift_range=0.10,
            shear_range=0.10)
            
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ]
        
        print('Begin training...')
        model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0]*5,
                            nb_epoch=nb_epoch,
                            validation_data=(X_valid, Y_valid),
                            callbacks=callbacks)

        print('Begin predictions...')
        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        yfull_test.append(test_prediction)

    predictions_valid = get_validation_predictions(train_data, yfull_train)
    score = log_loss(train_target, predictions_valid)
    print("Log_loss train independent avg: ", score)

    print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    perc = getPredScorePercent(train_target, train_id, predictions_valid)
    print('Percent success: {}'.format(perc))

    info_string = 'loss_' + str(score) \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    run_cross_validation(10)