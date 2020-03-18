import argparse
from utils import *
import tensorflow as tf
import time
from utils import *
from ops import *
from test import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.01)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.01, shape=shape)
#     return tf.Variable(initial)
#
# def dense(x,in_dim,out_dim):
#     weights = weight_variable([in_dim, out_dim])
#     bias = bias_variable([out_dim])
#     out = tf.add(tf.matmul(x,weights), bias)
#     return out

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of prototype generating network for ZSL"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir',type=str, default='CUB', help='[AwA1 / AwA2 / CUB / FLO]')
    parser.add_argument('--img_dim', type=int, default=2048, help='the image dimension')
    parser.add_argument('--hid_dim', type=int, default=1800, help='the hidden dimension, default: 1600')
    parser.add_argument('--mid_dim', type=int, default=1600, help='the middle dimension of discriminator, default: 1800')
    parser.add_argument('--att_dim', type=int, default=1024, help='the attribute dimension, AwA: 85, CUB: 1024,FLO: 1024')
    parser.add_argument('--cla_num', type=int, default=200, help='the class number')
    parser.add_argument('--tr_cla_num', type=int, default=150, help='the training class number')
    parser.add_argument('--selected_cla_num',type=int, default=10, help='the selected class number for meta-test')
    parser.add_argument('--lr', type=float32, default=5e-5, help='the learning rate, default: 1e-4')
    parser.add_argument('--preprocess', action='store_true', default=False, help='MaxMin process')
    parser.add_argument('--dropout',action='store_true',default=False, help='enable dropout')
    parser.add_argument('--epoch', type=int, default=15, help='the max iterations, default: 5000')
    parser.add_argument('--episode',type=int, default=100, help='the max iterations of episodes')
    parser.add_argument('--inner_loop',type=int, default=20, help='the inner loop')
    parser.add_argument('--batch_size', type=int, default=20, help='the batch_size, default: 100')
    parser.add_argument('--manualSeed', type=int, default=4198, help='maunal seed') # 4198
    return parser.parse_args()

class Model(object):
    def __init__(self,sess, args):
        self.args = args
        self.sess = sess
        self.att_dim = args.att_dim
        self.img_dim = args.img_dim
        self.hid_dim = args.hid_dim
        self.mid_dim = args.mid_dim
        self.cla_num = args.cla_num
        self.tr_cla_num = args.tr_cla_num
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.episode = args.episode
        self.dropout = args.dropout
        self.inner_loop = args.inner_loop
        self.lr = args.lr
        self.data = load_data(self.args)

        print("###### Information #######")
        print('# batch_size:', self.batch_size)
        print('# epoch_number:', self.epoch)
        print('# episde:', self.episode)
        print('# selected_cla_number:', args.selected_cla_num)
        print('# learning rate:', self.lr)
        print('# inner_loop:', self.inner_loop)
        self.create_model()

    ##################################################################################
    # Model
    ##################################################################################

    def create_model(self):

        self.img = tf.placeholder(tf.float32,[None,self.img_dim])
        self.att = tf.placeholder(tf.float32,[None,self.att_dim])
        self.cla = tf.placeholder(tf.float32,[None,self.tr_cla_num])
        self.att_pro = tf.placeholder(tf.float32, [None, self.att_dim])
        self.learn_rate = tf.placeholder(tf.float32)

        self.learner_img = tf.placeholder(tf.float32, [None, self.img_dim])
        self.learner_pro = tf.placeholder(tf.float32, [None, self.att_dim])
        self.learner_cla = tf.placeholder(tf.float32, [None, self.tr_cla_num])

        # Model parameter
        # F network
        self.gen_w1 = weight_variable([self.att_dim, self.hid_dim])
        self.gen_b1 = bias_variable([self.hid_dim])
        self.gen_w2 = weight_variable([self.hid_dim, self.img_dim])
        self.gen_b2 = bias_variable([self.img_dim])

        #
        self.pre_img = self.G(self.att)
        self.pre_att = self.F(self.img)

        # classification loss
        self.att_output_pro = self.G(self.att_pro)
        logit_img = tf.matmul(self.img, tf.transpose(self.att_output_pro))
        logit_att = tf.matmul(self.pre_att, tf.transpose(self.att_pro))
        cla_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_img, labels=self.cla)+
                                  tf.nn.softmax_cross_entropy_with_logits(logits=logit_att, labels=self.cla))

        # discriminative
        d_image_real = self.D(self.img, self.att)
        d_image_fake = self.D(self.pre_img, self.pre_att)

        self.d_loss = discriminator_loss('wgan',d_image_real, d_image_fake)
        mse = tf.reduce_sum((self.img - self.pre_img) ** 2, 1)
        e_loss = tf.reduce_mean(1e-3*mse + 1e-3*tf.log(d_image_real))
        self.g_loss = generator_loss('wgan',d_image_real, d_image_fake) + tf.log(e_loss)

        learner_pro_img = self.G(self.learner_pro)
        dists = euclidean_distance(learner_pro_img, self.learner_img)
        log_p_y = tf.nn.log_softmax(-dists)
        self.c_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(self.learner_cla, tf.transpose(log_p_y)), axis=-1),[-1]))

        lse_loss = tf.reduce_mean(tf.square(self.img - self.pre_img)) + tf.reduce_mean(tf.square(self.att - self.pre_att))
        # base loss
        self.b_loss = lse_loss + cla_loss

        ## Optimizer
        self.b_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.b_loss)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.d_loss)
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.g_loss)
        self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr*0.1).minimize(self.c_loss)

    def train(self):
        self.init = tf.global_variables_initializer()
        start_time = time.time()
        self.sess.run(self.init)

        for epo in range(self.epoch):
            meta_data, learner_data = prepare_data(self.data,self.args)
            for epi in range(self.episode):
                learner_fea, learner_pro, learner_lab = get_learner_data(learner_data)
                train_loader = get_batch(meta_data, self.batch_size)

                img_batch, att_batch, cla_batch, train_pro = train_loader.next()
                _, loss = self.sess.run([self.b_optimizer, self.b_loss],
                                            feed_dict={self.att:att_batch, self.img:img_batch,self.att_pro:train_pro,
                                                       self.cla:cla_batch, self.learner_img: learner_fea,
                                                       self.learner_pro: learner_pro, self.learner_cla: learner_lab})
                self.sess.run([self.d_optimizer],
                                  feed_dict={self.att: att_batch, self.img: img_batch})
                self.sess.run([self.g_optimizer],
                                  feed_dict={self.att: att_batch, self.img: img_batch})
                if (epi+1)%50 == 0:
                    print ('[epoch {}/{}, episode {}/{}] => loss:{:.5f}'.format(epo+1, self.epoch, epi+1, self.episode, loss))
            for i in range(self.inner_loop):
                learner_fea, learner_pro, learner_lab = get_learner_data(learner_data)
                self.sess.run(self.c_optimizer, feed_dict={self.learner_img:learner_fea, self.learner_pro:learner_pro, self.learner_cla:learner_lab})
           
            self.test()

        end_time = time.time()
        print('# The training time is: %4.4f' %(end_time - start_time))

    def test(self):
        start_time = time.time()
        nn = Test_nn(self.sess, self.att, self.pre_img, self.cla_num)

        acc = nn.test_zsl(self.data)
        print('Accuracy: unseen class', acc)

        acc_seen = nn.test_seen(self.data)
        acc_unseen = nn.test_unseen(self.data)
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print ('accuracy: unseen class:', acc_unseen, '| seen class:', acc_seen, '| harmonic:', H)

        end_time = time.time()
        print('# The test time is: %4.4f' %(end_time - start_time))

    def F(self, x):
        # img-->att, project the visual feature into the attribute space, g(x)
        with tf.name_scope('embedding'):
            hidden = tf.nn.tanh(dense(x, self.img_dim, self.hid_dim)) # tanh
            if self.dropout:
               hidden = tf.nn.dropout(hidden, 0.8)
            output = relu(dense(hidden, self.hid_dim, self.att_dim))
            return output

    def G(self,x, reuse=False):
        # att-->img, like a generation network f(x)
        # with tf.name_scope('generator'):
        with tf.variable_scope('generator',reuse=reuse):
            hidden = tf.nn.tanh(tf.matmul(x, self.gen_w1)+self.gen_b1) #
            output = relu(tf.matmul(hidden, self.gen_w2)+self.gen_b2)
        return output

    def D(self,x, c):
        with tf.name_scope('Discriminator'):
            inputs = tf.concat(axis=1, values=[x, c])
            middle = relu(dense(inputs,self.img_dim+self.att_dim,self.mid_dim))
            if self.dropout:
                middle = tf.nn.dropout(middle, 0.6) # 0.6
            output = relu(dense(middle,self.mid_dim,1))
            return output

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    random.seed(args.manualSeed)
    tf.set_random_seed(args.manualSeed)
    sess = tf.Session()

#    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    gan = Model(sess, args)
    #gan.create_model()
    gan.train()
    print(" [*] Training finished!")
    gan.test()
    print(" [*] Test finished!")
    sess.close()

if __name__ == '__main__':
    main()

