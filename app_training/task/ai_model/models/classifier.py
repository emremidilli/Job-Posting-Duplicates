import tensorflow as tf


class JobAddClassifier(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        ...
