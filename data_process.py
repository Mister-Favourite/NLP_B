import math
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import tensorflow_hub as hub

df_list = []

for i in range(1, 29):
    df_list.append(pd.read_csv('Data/Data '+str(i)+'.csv', sep=','))


print('14')
#Keras processing
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1")

print('19')
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1", trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]

print('26')
embedding_model = tf.keras.Model(text_input, pooled_output)
sentences = tf.constant([""])
print(embedding_model(sentences))