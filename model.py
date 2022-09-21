import numpy as np
import os

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.text_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

spec = model_spec.get('average_word_vec')
spec.num_words = 2000
spec.seq_len = 20
spec.wordvec_dim = 7

data = DataLoader.from_csv(
      filename='./spam.csv',
      text_column='text', 
      label_column='spam', 
      model_spec=spec,
      delimiter=',',
      shuffle=True,
      is_training=True)

train_data, test_data = data.split(0.9)

model = text_classifier.create(train_data, model_spec=spec, epochs=5)

model.export(export_dir="./JS/SPAM_DETECT_MODEL/", export_format=[
  ExportFormat.TFJS, 
  ExportFormat.LABEL, 
  ExportFormat.VOCAB
])