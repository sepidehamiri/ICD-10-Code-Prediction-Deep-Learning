import pymlconf
from appdirs import AppDirs

settings = pymlconf.DeferredConfigManager()

BUILTIN = '''
word2vector_filename: %(data_dir)s/hamster_data/word_vectors/word2vector.txt
dataset_filename: %(data_dir)s/hamster_data/dataset/dataset.csv
stop_words_filename: %(data_dir)s/hamster_data/stop_words/stopwords.txt
deep_learning_filename: %(data_dir)s/hamster_data/deep_model

machine_learning:
  labels: 19
  sequence_length: 100
  hidden_size: 150
  attention_size: 50
  probability: 0.8
  batch_size: 128
  epochs: 5
  delta: 0.5

dataset_split:
  test_size: 0.2
  validation_size: 0.2
  random_state: 1
  
''' % dict(
    data_dir=AppDirs().user_data_dir
)


def configure(*args, **kwargs):
    settings.load(*args, init_value=BUILTIN, **kwargs)
