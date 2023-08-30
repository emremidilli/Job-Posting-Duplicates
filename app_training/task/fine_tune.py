from ai_model.models import JobAddClassifier

import numpy as np

import os

import tensorflow as tf

TRAINING_DATA_DIR = '../bin-job-posting/training_data'
HF_PT_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MINI_BATCH_SIZE = 32


def get_with_major_soc_codes(X, y):
    '''
    choose only the job posts where SOC Code exists at least 10 times.
    '''
    a, inverse, count = np.unique(y, return_counts=True, return_inverse=True)
    a = a[count > 10]
    msk = np.isin(y, a)
    X = X[msk]
    y = inverse[msk]

    return X, y


if __name__ == '__main__':

    X_train = np.load(os.path.join(TRAINING_DATA_DIR, 'X_train.npy'),
                      allow_pickle=True)
    y_train = np.load(os.path.join(TRAINING_DATA_DIR, 'y_train.npy'),
                      allow_pickle=True)

    X_train, y_train = get_with_major_soc_codes(X_train, y_train)

    ds_train = tf.data.Dataset.from_tensor_slices(
        [X_train, y_train]).batch(
            MINI_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    classifier = JobAddClassifier(
        pre_trained_model_id=HF_PT_MODEL_ID,
        nr_of_classes=10
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-5)

    classifier.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]
    )

    classifier.fit(
        ds_train,
        epochs=5,
        verbose=2,
        shuffle=False)
