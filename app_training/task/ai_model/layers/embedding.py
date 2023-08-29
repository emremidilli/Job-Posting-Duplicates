import tensorflow as tf

from transformers import TFAutoModel


class Embedding(tf.keras.layers.Layer):
    '''Word embedding layer from HuggingFace'''
    def __init__(
            self,
            pre_trained_model_id='sentence-transformers/all-MiniLM-L6-v2',
            **kwargs):

        super().__init__(**kwargs)

        self.model = TFAutoModel.from_pretrained(pre_trained_model_id)

    def call(self, x):
        y = self.model(x)
        return y
