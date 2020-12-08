from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re


from .utils import batch_generator


class MachineLearning:
    def __init__(self, number_of_labels, label_orders, embedding_matrix, sequence_length=100, hidden_size=150,
                 attention_size=50, keep_probability=0.8, batch_size=128, number_of_epochs=5,
                 delta=0.5, save_path=''):

        self.labels = number_of_labels
        self.label_orders = label_orders
        self.sequence_length = sequence_length
        self.embedding_matrix = embedding_matrix
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.delta = delta
        self.save_path = save_path

        # Different placeholders
        with tf.name_scope('Inputs'):
            self.batch_ph = tf.placeholder(tf.int32, [None, self.sequence_length],
                                           name='batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            tf.summary.histogram('embeddings_var',
                                 self.embedding_matrix)
            self.batch_embedded = tf.nn.embedding_lookup(self.embedding_matrix,
                                                         self.batch_ph)

        # (Bi-)RNN layer(-s)
        self.rnn_outputs, _ = bi_rnn(GRUCell(self.hidden_size, reuse=tf.AUTO_REUSE),
                                     GRUCell(self.hidden_size, reuse=tf.AUTO_REUSE),
                                     inputs=self.batch_embedded,
                                     sequence_length=self.seq_len_ph, dtype=tf.float32)
        tf.summary.histogram('RNN_outputs', self.rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            self.inputs = tf.concat(self.rnn_outputs, 2)
            self.hidden_size_attention = self.inputs.shape[2].value
            self.w_omega = tf.Variable(tf.random_normal([self.hidden_size_attention,
                                                         self.attention_size], stddev=0.1))
            self.b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            self.u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                self.v = tf.tanh(tf.tensordot(self.inputs, self.w_omega, axes=1) + self.b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            self.vu = tf.tensordot(self.v, self.u_omega, axes=1)   # (B,T) shape
            self.alphas = tf.nn.softmax(self.vu)

            self.attention_output = tf.reduce_sum(self.inputs * tf.expand_dims(self.alphas, -1), 1)
            tf.summary.histogram('alphas', self.alphas)

        # Dropout
        self.drop = tf.nn.dropout(self.attention_output, self.keep_prob_ph)

        # Fully connected layer
        with tf.name_scope('Fully_connected_layer'):
            self.w = tf.Variable(tf.truncated_normal([self.hidden_size * 2, 1], stddev=0.1))
            self.b = tf.Variable(tf.constant(0., shape=[1]))
            self.y_hat = tf.nn.xw_plus_b(self.drop, self.w, self.b)
            self.y_hat = tf.squeeze(self.y_hat)
            tf.summary.histogram('W', self.w)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat,
                                                                               labels=self.target_ph))
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

            # Accuracy metric
            self.output = tf.sigmoid(self.y_hat)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.output),
                                                            self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./logdir/train'
                                                  , self.accuracy.graph)
        self.test_writer = tf.summary.FileWriter('./logdir/test'
                                                 , self.accuracy.graph)

        self.session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

        self.saver = tf.train.Saver(max_to_keep=self.labels)

    def train(self, x_train, y_train, x_test, y_test):
        x_test_all = np.array(x_test)
        x_train_all = np.array(x_train)
        y_train_all = np.array(y_train)
        y_test_all = np.array(y_test)
        for index in range(0, self.labels):
            y_train = y_train_all[:, index]
            y_test = y_test_all[:, index]

            # Batch generators
            train_batch_generator = batch_generator(x_train_all,
                                                    y_train, self.batch_size)
            test_batch_generator = batch_generator(x_test_all,
                                                   y_test, self.batch_size)
            with tf.Session(config=self.session_conf) as sess:
                sess.run(tf.global_variables_initializer())
                print("Start learning...")
                for epoch in range(self.number_of_epochs):
                    loss_train = 0
                    loss_test = 0
                    accuracy_train = 0
                    accuracy_test = 0

                    print("epoch: {}\t".format(epoch), end="")

                    # Training
                    num_batches = x_train_all.shape[0] // self.batch_size
                    for b in tqdm(range(num_batches)):
                        x_batch, y_batch = next(train_batch_generator)
                        seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                        loss_tr, acc, _, summary = sess.run([self.loss, self.accuracy,
                                                             self.optimizer, self.merged],
                                                            feed_dict={self.batch_ph: x_batch,
                                                                       self.target_ph: y_batch,
                                                                       self.seq_len_ph: seq_len,
                                                                       self.keep_prob_ph: self.keep_probability})
                        accuracy_train += acc
                        loss_train = loss_tr * self.delta + loss_train * (1 - self.delta)
                        self.train_writer.add_summary(summary, b + num_batches * epoch)
                    accuracy_train /= num_batches

                    # Testing
                    num_batches = x_test_all.shape[0] // self.batch_size
                    for b in tqdm(range(num_batches)):
                        x_batch, y_batch = next(test_batch_generator)
                        seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                        loss_test_batch, acc, summary = sess.run([self.loss, self.accuracy, self.merged],
                                                                 feed_dict={self.batch_ph: x_batch,
                                                                            self.target_ph: y_batch,
                                                                            self.seq_len_ph: seq_len,
                                                                            self.keep_prob_ph: 1.0})
                        accuracy_test += acc
                        loss_test += loss_test_batch
                        self.test_writer.add_summary(summary, b + num_batches * epoch)
                    accuracy_test /= num_batches
                    loss_test /= num_batches

                    print('loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}'.format(
                        loss_train, loss_test, accuracy_train, accuracy_test
                    ))
                self.train_writer.close()
                self.test_writer.close()
                self.saver.save(sess, '{}model{}'.format(self.save_path, index))
            print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")

    def predict_labels(self, x_seq, text, threshold_label, threshold_alpha):
        output = []
        threshold_alpha = np.float32(threshold_alpha)
        name = ['code', 'accuracy', 'tokens']
        indices = []
        words = text.split()
        for j, word in enumerate(words):
            q = re.sub('[^[A-Za-z]', '', word)
            if q != '':
                indices.append(j)
        for i in range(0, self.labels):
            result = dict()
            tokens = []
            # Calculate alpha coefficients for the first test example
            with tf.Session() as sess:
                self.saver.restore(sess, '{}model{}'.format(self.save_path, i))
                x_batch_test, y_batch_test = x_seq, [1]
                seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])

                y = sess.run([self.output], feed_dict={self.batch_ph: x_batch_test,
                                                       self.target_ph: y_batch_test,
                                                       self.seq_len_ph: seq_len_test,
                                                       self.keep_prob_ph: 1.0})
                alphas_test = sess.run([self.alphas], feed_dict={self.batch_ph: x_batch_test,
                                                                 self.target_ph: y_batch_test,
                                                                 self.seq_len_ph: seq_len_test,
                                                                 self.keep_prob_ph: 1.0})

            alphas_values = alphas_test[0][0]
            if y[0] >= threshold_label:
                code = str(self.label_orders[i])
                accuracy = y[0]
                alphas = np.zeros(len(words))
                with open('visualization.html', 'a') as html_file:
                    html_file.write('<p><font style="background: rgba(255, 255, 0, 0)'
                                    '">code: %s</font> </p>\n' % code)
                    for index, token in enumerate(zip(indices,
                                                      alphas_values / alphas_values.max())):
                        if token[1] >= threshold_alpha:
                            alpha = token[1]
                            tokens.append(dict([(str(indices[index]), float(alpha))]))
                            alphas[indices[index]] = alpha

                    for index, word in enumerate(words):
                        html_file.write('<font style="background: rgba(255, 255, 0, %f)"'
                                        '>%s</font>\n' % (alphas[index], word))
                result[name[0]] = code
                result[name[1]] = float(accuracy)
                result[name[2]] = tokens
                output.append(result)
            sess.close()
        return output
