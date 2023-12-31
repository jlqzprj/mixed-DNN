
import tensorflow as tf
import os
print(tf.__version__)
import numpy as np

import scipy.io
import time
from matplotlib import pyplot as plt


np.random.seed(1234)
tf.set_random_seed(1234)


class kohnint:
    def __init__(self, X_quad, X_full, X_bdr, layers1, layers2, theta, normalx, normaly, normalz, f, g, dudxq, dudyq,
                 dudzq):

        self.theta = theta
        
        self.xq = X_quad[:, 0:1]
        self.yq = X_quad[:, 1:2]
        self.zq = X_quad[:, 2:3]
        
        self.xf = X_full[:, 0:1]
        self.yf = X_full[:, 1:2]
        self.zf = X_full[:, 2:3]
        
        self.normalx = normalx
        self.normaly = normaly
        self.normalz = normalz
        
        self.xb = X_bdr[:, 0:1]
        self.yb = X_bdr[:, 1:2]
        self.zb = X_bdr[:, 2:3]
        
        self.f = f
        self.g = g
 
        self.layers1 = layers1
        self.layers2 = layers2
        
        self.dudxq = dudxq
        self.dudyq = dudyq
        self.dudzq = dudzq

        # Initialize trainable parameters weights and biases
        self.weights1, self.biases1 = self.initialize_NN(self.layers1)
        self.weights2, self.biases2 = self.initialize_NN(self.layers2)

        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))


        # placeholders were created to ease batch training
        self.xq_tf = tf.placeholder(tf.float32, shape=[None, self.xq.shape[1]])
        self.yq_tf = tf.placeholder(tf.float32, shape=[None, self.yq.shape[1]])
        self.zq_tf = tf.placeholder(tf.float32, shape=[None, self.zq.shape[1]])
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.yf_tf = tf.placeholder(tf.float32, shape=[None, self.yf.shape[1]])
        self.zf_tf = tf.placeholder(tf.float32, shape=[None, self.zf.shape[1]])
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.yb_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.zb_tf = tf.placeholder(tf.float32, shape=[None, self.zb.shape[1]])
        self.nx_tf = tf.placeholder(tf.float32, shape=[None, self.normalx.shape[1]])
        self.ny_tf = tf.placeholder(tf.float32, shape=[None, self.normaly.shape[1]])
        self.nz_tf = tf.placeholder(tf.float32, shape=[None, self.normalz.shape[1]])
        self.f_tf = tf.placeholder(tf.float32, shape=[None, self.f.shape[1]])
        self.g_tf = tf.placeholder(tf.float32, shape=[None, self.g.shape[1]])
        self.dudxq_tf = tf.placeholder(tf.float32, shape=[None, self.dudxq.shape[1]])
        self.dudyq_tf = tf.placeholder(tf.float32, shape=[None, self.dudyq.shape[1]])
        self.dudzq_tf = tf.placeholder(tf.float32, shape=[None, self.dudzq.shape[1]])


        # net_a here is the network for the vector field sigma in the paper
        self.a = self.net_a(self.xq_tf, self.yq_tf, self.zq_tf)
        # net_sigma here is the network for the conductivity q in the paper
        # xq,yq,zq are the collocation points sampled from the subdomain w
        self.sig,_,_,_ = self.net_sigma(self.xq_tf, self.yq_tf, self.zq_tf)
        # xf,yf,zf are the collocation points sampled from the whole domain Omega
        self.sigf, self.sigxf, self.sigyf, self.sigzf = self.net_sigma(self.xf_tf, self.yf_tf, self.zf_tf)
        self.abdr = self.net_a(self.xb_tf, self.yb_tf, self.zb_tf)
        self.diva = self.div1(self.xf_tf, self.yf_tf, self.zf_tf)
        
        
        ################################################################
        ########################## IMPORTANT ###########################
        ################################################################
        
        # sigprj is the projected conductivity
        # for detailed instructions on using self.sigprj in the loss, refer to readme.md section 2 (initialization)
        self.c0 = 0.1 * tf.ones([self.xq.shape[0], 1])
        self.c1 = 10 * tf.ones([self.xq.shape[0], 1])
        self.sigprj = tf.math.minimum(tf.math.maximum(self.c0, self.sig), self.c1)
        
        # the max and min value of the conductivity q
        self.s1 = tf.reduce_max(self.sigf)
        self.s0 = tf.reduce_min(self.sigf)


        # the loss function
        self.loss1 = 0.784 * tf.reduce_mean(tf.square(self.a[:, 0:1] - self.sigprj * self.dudxq_tf) + tf.square(
            self.a[:, 1:2] - self.sigprj * self.dudyq_tf) + tf.square(self.a[:, 2:3] - self.sigprj * self.dudzq_tf))
        
        self.loss2 = 10 * tf.reduce_mean(tf.square(self.diva + self.f_tf))
        
        self.loss3 = 60 * tf.reduce_mean(tf.square(
            self.abdr[:, 0:1] * self.nx_tf + self.abdr[:, 1:2] * self.ny_tf + self.abdr[:,
                                                                              2:3] * self.nz_tf - self.g_tf))
        
        #self.loss4 = tf.reduce_mean(tf.square(self.sig))
        
        self.loss5 = tf.reduce_mean(tf.square(self.sigxf) + tf.square(self.sigyf) + tf.square(self.sigzf))

        self.loss6 = 0.00001 * (self.loss5)

        self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss6


        # initialize empty vct to store the loss values after each epoch for visualizing the loss
        self.loss_vals = []
        self.loss_vals1 = []
        self.loss_vals2 = []
        self.loss_vals3 = []
        self.loss_vals6 = []
        
        
        # optimizer
        global_step = tf.Variable(0, trainable=False)
        start_learning_rate = 0.003
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 3000, 0.8)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                                                                        global_step=global_step)


        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # basic feed forward architecture and initialization, see reference [3] in readme.md
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], 0, stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    
    # huber smoothing for the square root function, used only in ex5.2 & 5.8
    def huber(self, theta, x1, x2):
        thetavec = theta * tf.ones(tf.shape(x1))
        eucnorm = tf.square(x1) + tf.square(x2)
        condition = tf.greater_equal(eucnorm, thetavec ** 2)
        large_res = tf.sqrt(eucnorm)
        small_res = 0.5 * 1. / thetavec * eucnorm + 0.5 * thetavec
        return tf.where(condition, large_res, small_res)



    def net_a(self, x, y, z):
        a = self.neural_net(tf.concat([x, y, z], 1), self.weights2, self.biases2)
        return a

    def net_sigma(self, x, y, z):
        sig = self.neural_net(tf.concat([x, y, z], 1), self.weights1, self.biases1)
        gradsigx = tf.gradients(sig, x)[0]
        gradsigy = tf.gradients(sig, y)[0]
        gradsigz = tf.gradients(sig, z)[0]
        return sig, gradsigx, gradsigy, gradsigz

    def div1(self, x, y, z):
        a = self.net_a(x, y, z)
        ax = a[:, 0:1]
        ay = a[:, 1:2]
        az = a[:, 2:3]
        gradax = tf.gradients(ax, x)[0]
        graday = tf.gradients(ay, y)[0]
        gradaz = tf.gradients(az, z)[0]
        div = gradax + graday + gradaz
        return div



    def train(self):

        tf_dict = {self.xq_tf: self.xq, self.yq_tf: self.yq, self.zq_tf: self.zq, self.xb_tf: self.xb,
                   self.yb_tf: self.yb, self.zb_tf: self.zb, self.f_tf: self.f, self.g_tf: self.g,
                   self.nx_tf: self.normalx, self.ny_tf: self.normaly, self.nz_tf: self.normalz,
                   self.dudxq_tf: self.dudxq, self.dudyq_tf: self.dudyq, self.dudzq_tf: self.dudzq,
                   self.xf_tf: self.xf, self.yf_tf: self.yf, self.zf_tf: self.zf}
        
        
        #########################################################
        ####################### IMPORTANT #######################
        #########################################################
        
        # check readme.md section 2 for saving and restoring initializations of weights and biases
        
        #saver = tf.train.Saver(tf.trainable_variables())
        #modeltoRestore = os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE.ckpt')
        #saver.restore(self.sess, modeltoRestore)

        for epoch in range(70000):
            self.sess.run(self.optimizer, feed_dict=tf_dict)
            iter_loss1 = self.sess.run(self.loss1, feed_dict=tf_dict)
            iter_loss2 = self.sess.run(self.loss2, feed_dict=tf_dict)
            iter_loss3 = self.sess.run(self.loss3, feed_dict=tf_dict)
            iter_loss6 = self.sess.run(self.loss6, feed_dict=tf_dict)
            iter_loss = self.sess.run(self.loss, feed_dict=tf_dict)
            # print out s0 and s1 to see when q_theta (i.e. self.sigma) is well initialized
            s0 = self.sess.run(self.s0, feed_dict=tf_dict)
            s1 = self.sess.run(self.s1, feed_dict=tf_dict)
            
            if epoch % 100 == 0:
                print("Iter " + str(epoch) + ", loss = " + str(iter_loss))
                print("\t\tloss1 = " + str(1 * iter_loss1))
                print("\t\tloss2 = " + str(1 * iter_loss2))
                print("\t\tloss3 = " + str(1 * iter_loss3))
                print("\t\tloss6 = " + str(1 * iter_loss6))
                print("Iter " + str(epoch) + ", s1 = " + str(s1) + ", s0 = " + str(s0))
                
            self.loss_vals.append(iter_loss)
            self.loss_vals1.append(1 * iter_loss1)
            self.loss_vals2.append(1 * iter_loss2)
            self.loss_vals3.append(1 * iter_loss3)
            self.loss_vals6.append(1 * iter_loss6)
            
        #save_path = saver.save(self.sess, os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE.ckpt'))



    def predict(self, X_star):
        sigma = self.sess.run(self.sigf,
                              {self.xf_tf: X_star[:, 0:1], self.yf_tf: X_star[:, 1:2], self.zf_tf: X_star[:, 2:3]})
        return sigma


if __name__ == "__main__":
    
    # huber constant for smoothing the square root function, used only in ex5.2 & ex5.8
    theta = 0.000001

    # feed forward NN architecture
    layers1 = [3, 26, 26, 26, 10, 1]
    layers2 = [3, 26, 26, 26, 10, 3]

    # load data
    #data = scipy.io.loadmat('YOUR_PATH/neupartial3dnewsampparan10.mat')
    data = scipy.io.loadmat('YOUR_PATH/neupartial3d.mat')
    X_quad = data['X_quad']
    X_bdr = data['X_bdr']
    X_star = data['X_pred']
    X_full = data['X_quadfull']
    f = data['f']
    g = data['g']
    nx = data['normalx']
    ny = data['normaly']
    nz = data['normalz']
    dudxq = data['gux1quad']
    dudyq = data['gux2quad']
    dudzq = data['gux3quad']
    sigma_eval1 = data['sigma_eval1']

    # model input
    model = kohnint(X_quad, X_full, X_bdr, layers1, layers2, theta, nx, ny, nz, f, g, dudxq, dudyq, dudzq)
    model.X_star = X_star

    # train the model
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    # make the prediction
    sigpred = model.predict(X_star)

    # evaluate the L2-relative error
    error_sigma = np.linalg.norm(sigpred - sigma_eval1, 2) / np.linalg.norm(sigma_eval1, 2)
    print('Error sigma: %e' % (error_sigma))
    
    # save the conductivity values at selected prediction points in .mat file
    scipy.io.savemat('YOUR_OUTPUT_FILE_NAME.mat', {'sigma': sigpred})

    
    # plot the loss
    plt.figure(1)
    x1 = np.arange(1, len(model.loss_vals) + 1)
    plt.semilogy(x1, model.loss_vals, label='l')
    plt.semilogy(x1, model.loss_vals1, label='1')
    plt.semilogy(x1, model.loss_vals2, label='2')
    plt.semilogy(x1, model.loss_vals3, label='3')
    plt.semilogy(x1, model.loss_vals6, label='6')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.ylim(1e-5, 1500)
    plt.legend(loc=0, ncol=2)
    plt.show()
    
