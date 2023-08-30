from ai_model.layers import Embedding, Tokenizer

import tensorflow as tf


class JobAddClassifier(tf.keras.Model):
    '''Classifier from job embeddings to SOC code.'''
    def __init__(
            self,
            pre_trained_model_id,
            nr_of_classes,
            **kwargs):
        super().__init__(**kwargs)

        self.nr_of_classes = nr_of_classes

        self.tokenizer = Tokenizer(pre_trained_model_id=pre_trained_model_id)

        self.embedding = Embedding(pre_trained_model_id)

        self.dense = tf.keras.layers.Dense(
            units=self.nr_of_classes,
            activation='softmax')

    def call(self, x):
        '''
        x: tokenized input. (None, max_seq_lenth)
        y: class (None, 1)
        '''
        x = self.tokenizer(x)
        y = self.embedding(x)
        y = self.dense(y)

        return y
