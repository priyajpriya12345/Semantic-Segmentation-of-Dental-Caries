import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from sklearn.utils import shuffle

from Evaluation import net_evaluation


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def Model_ResUnetPlusPlus(Path, Mask, save_path='predict', model_name='model.h5', sol=None):
    if sol is None:
        sol = [1e-5, 1]
    tf.random.set_seed(42)
    np.random.seed(42)

    ## Path
    file_path = "files/"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = Path + "train/"
    valid_path = Path + "valid/"

    ## Training
    train_image_paths = sorted(glob(os.path.join(train_path, "image*.png")))
    train_mask_paths = sorted(glob(os.path.join(train_path, "mask*.png")))

    ## Shuffling
    train_image_paths, train_mask_paths = shuffling(train_image_paths, train_mask_paths)

    ## Validation
    valid_image_paths = sorted(glob(os.path.join(valid_path, "image*.png")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask*.png")))

    ## Parameters
    image_size = 256
    batch_size = 16
    lr = sol[0]
    epochs = int(sol[1])
    model_path = "files/resunetplusplus.h5"

    train_dataset = tf_dataset(train_image_paths, train_mask_paths)
    valid_dataset = tf_dataset(valid_image_paths, valid_mask_paths)

    try:
        arch = ResUnet(input_size=image_size)
        model = arch.build_model()
        model = tf.distribute.MirroredStrategy(model, 4, cpu_merge=False)
        print("Training using multiple GPUs..")
    except:
        arch = ResUnet(input_size=image_size)
        model = arch.build_model()
        print("Training using single GPU or CPU..")

    optimizer = Nadam(learning_rate=lr)
    # metrics = [dice_coef, MeanIoU(num_classes=2), Recall(), Precision()]
    metrics = [MeanIoU(num_classes=2), Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    model.summary()

    callbacks = [
        # ModelCheckpoint(model_path),
        #         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
        # schedule
    ]

    train_steps = (len(train_image_paths) // batch_size)
    valid_steps = (len(valid_image_paths) // batch_size)

    if len(train_image_paths) % batch_size != 0:
        train_steps += 1

    if len(valid_image_paths) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
              epochs=epochs,
              validation_data=valid_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=callbacks,
              shuffle=False)

    Images = []
    Target = []
    for i in range(len(valid_image_paths)):
        image = cv2.imread(valid_image_paths[i])
        mask = cv2.imread(valid_image_paths[i].replace("image", "mask"))
        pred_mask = tta_model(model, image)
        pred_mask = pred_mask.squeeze()
        cv2.imwrite(Path + save_path + '/image%04d.png' % (i + 1), pred_mask)
        Images.append(pred_mask)
        Target.append(mask)

    model.save(model_name)
    Eval = net_evaluation(Path, Mask)
    return Images, Eval