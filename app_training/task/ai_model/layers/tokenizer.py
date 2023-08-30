import tensorflow as tf

from transformers import AutoTokenizer


class Tokenizer(tf.keras.layers.Layer):
    '''Tokenizer layer from HuggingFace'''
    def __init__(
            self,
            pre_trained_model_id,
            **kwargs):

        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_id)

    def call(self, x):
        y = self.tokenizer(
            x,
            padding=True,
            truncation=True,
            return_tensors='tf')

        return y
