from time import strftime

import numpy as np
import gensim
import tensorflow as tf


def loading_w2v_model(word2vector_filename):

    print('Word2Vec loading start at: {}'.format(strftime('%H:%M:%S')))
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vector_filename, binary=False)
    return w2v_model


def create_embedding_matrix(model):

    embedding_matrix = np.zeros((len(model.vocab), model.vector_size), dtype=np.float32)
    for i in range(len(model.vocab)):
        embedding_vector = model[model.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_embedding_layer(embedding_matrix):
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        tensor = tf.constant(embedding_matrix,
                             dtype=tf.float32)
    embeddings_var = tf.Variable(tensor,
                                 trainable=False)
    return embeddings_var


def convert_data_to_index(w2v_model, documents):
    unknown_index = len(w2v_model.vocab)-1
    index_docs = []
    for doc in documents:
        index_word = []
        for word in doc[0].split():
            if word in w2v_model:
                index_word.append(w2v_model.vocab[word].index)
            else:
                index_word.append(unknown_index)
        index_docs.append(index_word)
    return index_docs
