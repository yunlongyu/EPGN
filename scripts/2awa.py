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
    parser.add_argument('--data_dir',type=str, default='AwA2', help='[AwA1 / AwA2 / CUB / FLO]')
    parser.add_argument('--img_dim', type=int, default=2048, help='the image dimension')
    parser.add_argument('--hid_dim', type=int, default=1800, help='the hidden dimension, default: 1600')
    parser.add_argument('--mid_dim', type=int, default=1600, help='the middle dimension of discriminator, default: 1800')
    parser.add_argument('--att_dim', type=int, default=85,help='the attribute dimension, AwA: 85, CUB: 1024, FLO: 1024')
    parser.add_argument('--cla_num', type=int, default=50, help='the class number')  # AwA: 50, CUB: 200, aPY: 32
    parser.add_argument('--tr_cla_num', type=int, default=40, help='the training class number')
    parser.add_argument('--selected_cla_num',type=int, default=10, help='the selected class number for meta-test')
    parser.add_argument('--lr', type=float32, default=1e-4, help='the learning rate, default: 1e-4')
    parser.add_argument('--preprocess', action='store_true', default=False, help='MaxMin process')
    parser.add_argument('--dropout',action='store_true',default=False,help='enable dropout layer')
    parser.add_argument('--epoch', type=int, default=30, help='the max iterations, default: 5000')
    parser.add_argument('--episode',type=int, default=100, help='the max iterations of episodes')
    parser.add_argument('--inner_loop',type=int, default=100, help='the inner loop')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch_size, default: 100')
    parser.add_argument('--manualSeed', type=int, default=4198, help='maunal seed') #8192
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
        self.inner_loop = args.inner_loop
        self.dropout = args.dropout
        self.lr = args.lr
        self.data = load_data(self.args)

        print("###### Information #######")
        print('# batch_size:', self.batch_size)
        print('# epoch_number:', self.epoch)
        print('# inner_loop number', self.inner_loop)
        print('# selected_class_number', args.selected_cla_num)
        print('# learning rate', self.lr)
        print('# manualSeed', args.manualSeed)
        self.create_model()

    ##################################################################################
    # Model
    ##################################################################################

    def create_model(self):
        self.img = tf.placeholder(tf.float32,[None,self.img_dim])
        self.att = tf.placeholder(tf.float32,[None,self.att_dim])
        self.cla = tf.placeholder(tf.float32,[None,self.tr_cla_num])
        self.pro = tf.placeholder(tf.float32, [None, self.att_dim])
        self.lr_pl = tf.placeholder(tf.float32)

        self.learner_img = tf.placeholder(tf.float32, [None, self.img_dim])
        self.learner_att = tf.placeholder(tf.float32, [None, self.att_dim])
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
        self.att_output_pro = self.G(self.pro,reuse=True)
        logit_img = tf.matmul(self.img, tf.transpose(self.att_output_pro))
        logit_att = tf.matmul(self.pre_att, tf.transpose(self.pro))
        cla_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_img, labels=self.cla) +
                                 tf.nn.softmax_cross_entropy_with_logits(logits=logit_att, labels=self.cla))

        # discriminative
        d_image_real = self.D(self.img, self.att)
        d_image_fake = self.D(self.pre_img, self.pre_att)

        self.d_loss = discriminator_loss('wgan',d_image_real, d_image_fake)
        mse = tf.reduce_sum((self.img - self.pre_img) ** 2, 1)
        e_loss = tf.reduce_mean(1e-3*mse + 1e-3*tf.log(d_image_real))
        self.g_loss = generator_loss('wgan',d_image_real, d_image_fake) + tf.log(e_loss)

        ## Learner loss
        learner_pro_img = self.G(self.learner_pro, reuse=True)
        dists = euclidean_distance(learner_pro_img, self.learner_img)
        log_p_y = tf.nn.log_softmax(-dists)
        self.c_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(self.learner_cla, tf.transpose(log_p_y)),axis=-1),[-1]))

        lse_loss = tf.reduce_mean(tf.square(self.img - self.pre_img)) + tf.reduce_mean(tf.square(self.att - self.pre_att))
        # base loss
        self.b_loss = lse_loss + cla_loss 

        ## Optimizer
        self.b_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_pl).minimize(self.b_loss)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_pl).minimize(self.d_loss)
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_pl).minimize(self.g_loss)
        self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_pl).minimize(self.c_loss)

    def train(self):
        best_acc = best_H = 0
        self.init = tf.global_variables_initializer()
        start_time = time.time()
        self.sess.run(self.init)

        for epo in range(self.epoch):
            meta_data, learner_data = prepare_data(self.data,self.args)
            for epi in range(self.episode):
                train_loader = get_batch(meta_data, self.batch_size)
                img_batch,att_batch, cla_batch, train_pro = train_loader.next()
                _, loss = self.sess.run([self.b_optimizer, self.b_loss],
                                            feed_dict={self.att:att_batch, self.img:img_batch,self.pro:train_pro,
                                                       self.cla:cla_batch, self.lr_pl:self.lr})
                self.sess.run([self.d_optimizer],
                                  feed_dict={self.att: att_batch, self.img: img_batch,self.lr_pl:self.lr})
                self.sess.run([self.g_optimizer],
                                  feed_dict={self.att: att_batch, self.img: img_batch,self.lr_pl:self.lr})

                if (epi+1)%50 == 0:
                   print ('[epoch {}/{}, episode {}/{}] => loss:{:.5f}'.format(epo+1, self.epoch, epi+1, self.episode, loss))
            for i in range(self.inner_loop):  #self.inner_loop
                learner_fea, learner_pro, learner_lab = get_learner_data(learner_data)
                self.sess.run(self.c_optimizer,feed_dict={self.learner_img:learner_fea,self.learner_pro:learner_pro, self.learner_cla:learner_lab,self.lr_pl:self.lr*0.1})

            acc, gzsl_acc = self.test()
            if acc > best_acc:
                best_acc = acc
            if gzsl_acc['H'] > best_H:
                best_H = gzsl_acc['H']

            best_gzsl={
                'acc_unseen':gzsl_acc['acc_unseen'],
                'acc_seen':gzsl_acc['acc_seen'],
                'H':gzsl_acc['H']
                }
        print ('Accuracy: unseen class:', best_acc)
        print ('Accuracy: unseen class:', best_gzsl['acc_unseen'],'| seen class:', best_gzsl['acc_seen'],'| harmonic:', best_gzsl['H'])

    def test(self):

        nn = Test_nn(self.sess, self.att, self.pre_img, self.cla_num)
        acc = nn.test_zsl(self.data)
        acc_seen = nn.test_seen(self.data)
        acc_unseen = nn.test_unseen(self.data)
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)	
        #print('Accuracy: unseen class:', acc_unseen,'| seen class:', acc_seen, '| harmonic:', H)
        gzsl_acc = {
            'acc_unseen':acc_unseen,
            'acc_seen':acc_seen,
            'H':H
            }
        return acc, gzsl_acc
    def F(self, x):
        # img-->att, project the visual feature into the attribute space, g(x)
        with tf.name_scope('embedding'):
            hidden = tf.nn.relu(dense(x, self.img_dim, self.hid_dim))
            if self.dropout:
                hidden = tf.nn.dropout(hidden, 0.8) #0.8
            output = tf.nn.relu(dense(hidden, self.hid_dim, self.att_dim))
            return output

    def G(self,x, reuse=False):
        # att-->img, like a generation network f(x)
        with tf.variable_scope('generator', reuse=reuse):
            hidden = tf.nn.tanh(tf.matmul(x, self.gen_w1)+self.gen_b1)
            output = tf.nn.relu(tf.matmul(hidden, self.gen_w2)+self.gen_b2)
            return output

    def D(self,x, c):
        with tf.name_scope('discriminator'):
            inputs = tf.concat(axis=1, values=[x, c])
            middle = tf.nn.relu(dense(inputs,self.img_dim+self.att_dim,self.mid_dim))
            if self.dropout:
                middle = tf.nn.dropout(middle,0.5) # 0.5
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

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = Model(sess, args)
        gan.train()

if __name__ == '__main__':
    main()

