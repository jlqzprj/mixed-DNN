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
    def __init__(self, X_quad, X_bdr, layers1, layers2, theta, f, g1, g2, g3, g4, g5,
                 dudx1q, dudx2q, dudx3q, dudx4q, dudx5q, sigbdr):
        
        self.theta = theta
        
        self.x1q = X_quad[:, 0:1]
        self.x2q = X_quad[:, 1:2]
        self.x3q = X_quad[:, 2:3]
        self.x4q = X_quad[:, 3:4]
        self.x5q = X_quad[:, 4:5]

        self.g1 = g1 #(g1,g2,g3,g4,g5) = q^dagger*\grad(z^delta) in the paper, corresponds to the dirichlet condition
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4
        self.g5 = g5

        self.x1b = X_bdr[:, 0:1]
        self.x2b = X_bdr[:, 1:2] 
        self.x3b = X_bdr[:, 2:3]
        self.x4b = X_bdr[:, 3:4]
        self.x5b = X_bdr[:, 4:5] 

        self.f = f

        self.layers1 = layers1
        self.layers2 = layers2
        
        self.dudx1q = dudx1q
        self.dudx2q = dudx2q
        self.dudx3q = dudx3q
        self.dudx4q = dudx4q
        self.dudx5q = dudx5q

        self.sigbdr = sigbdr


        # Initialize trainable parameters weights and biases
        self.weights1, self.biases1 = self.initialize_NN(self.layers1)
        self.weights2, self.biases2 = self.initialize_NN(self.layers2)

        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))


        # placeholders were created to ease batch training
        self.x1q_tf = tf.placeholder(tf.float32, shape=[None, self.x1q.shape[1]])
        self.x2q_tf = tf.placeholder(tf.float32, shape=[None, self.x2q.shape[1]])
        self.x3q_tf = tf.placeholder(tf.float32, shape=[None, self.x3q.shape[1]])
        self.x4q_tf = tf.placeholder(tf.float32, shape=[None, self.x4q.shape[1]])
        self.x5q_tf = tf.placeholder(tf.float32, shape=[None, self.x5q.shape[1]])

        self.x1b_tf = tf.placeholder(tf.float32, shape=[None, self.x1b.shape[1]])
        self.x2b_tf = tf.placeholder(tf.float32, shape=[None, self.x2b.shape[1]])
        self.x3b_tf = tf.placeholder(tf.float32, shape=[None, self.x3b.shape[1]])
        self.x4b_tf = tf.placeholder(tf.float32, shape=[None, self.x4b.shape[1]])
        self.x5b_tf = tf.placeholder(tf.float32, shape=[None, self.x5b.shape[1]])

        self.g1_tf = tf.placeholder(tf.float32, shape=[None, self.g1.shape[1]])
        self.g2_tf = tf.placeholder(tf.float32, shape=[None, self.g2.shape[1]])
        self.g3_tf = tf.placeholder(tf.float32, shape=[None, self.g3.shape[1]])
        self.g4_tf = tf.placeholder(tf.float32, shape=[None, self.g4.shape[1]])
        self.g5_tf = tf.placeholder(tf.float32, shape=[None, self.g5.shape[1]])

        self.f_tf = tf.placeholder(tf.float32, shape=[None, self.f.shape[1]])
        self.sigbdr_tf = tf.placeholder(tf.float32, shape=[None, self.sigbdr.shape[1]])

        self.dudx1q_tf = tf.placeholder(tf.float32, shape=[None, self.dudx1q.shape[1]])
        self.dudx2q_tf = tf.placeholder(tf.float32, shape=[None, self.dudx2q.shape[1]])
        self.dudx3q_tf = tf.placeholder(tf.float32, shape=[None, self.dudx3q.shape[1]])
        self.dudx4q_tf = tf.placeholder(tf.float32, shape=[None, self.dudx4q.shape[1]])
        self.dudx5q_tf = tf.placeholder(tf.float32, shape=[None, self.dudx5q.shape[1]])

        

        # net_a here is the network for the vector field sigma in the paper
        # (xq,yq,zq) are the collocation points sampled from the domain interior Omega
        self.a = self.net_a(self.x1q_tf, self.x2q_tf, self.x3q_tf, self.x4q_tf, self.x5q_tf)
        # net_sigma here is the network for the conductivity q in the paper
        self.sig, self.sigx1, self.sigx2, self.sigx3, self.sigx4, self.sigx5 = self.net_sigma(
            self.x1q_tf, self.x2q_tf, self.x3q_tf, self.x4q_tf, self.x5q_tf)
        
        self.abdr = self.net_a(self.x1b_tf, self.x2b_tf, self.x3b_tf, self.x4b_tf, self.x5b_tf)
        self.diva = self.div1(self.x1q_tf, self.x2q_tf, self.x3q_tf, self.x4q_tf, self.x5q_tf)
        self.sigb, _, _, _, _, _ = self.net_sigma(self.x1b_tf, self.x2b_tf, self.x3b_tf, self.x4b_tf, self.x5b_tf)
        
        
        
        ################################################################
        ########################## IMPORTANT ###########################
        ################################################################
        
        # sigprj is the projected conductivity
        # for detailed instructions on using self.sigprj in the loss, refer to readme.md section 2 (initialization)
        self.c0 = 0.1 * tf.ones([self.x1q.shape[0], 1])
        self.c1 = 10 * tf.ones([self.x1q.shape[0], 1])
        self.sigprj = tf.math.minimum(tf.math.maximum(self.c0, self.sig), self.c1)
        
        # the max and min value of the conductivity q
        self.s1 = tf.reduce_max(self.sig)
        self.s0 = tf.reduce_min(self.sig)
        
        
        
        # the loss function
        self.loss1 = 1 * tf.reduce_mean(tf.square(self.a[:, 0:1] - self.sigprj * self.dudx1q_tf) + tf.square(
            self.a[:, 1:2] - self.sigprj * self.dudx2q_tf) +
                                        tf.square(self.a[:, 2:3] - self.sigprj * self.dudx3q_tf) + tf.square(
            self.a[:, 3:4] - self.sigprj * self.dudx4q_tf) +
                                        tf.square(self.a[:, 4:5] - self.sigprj * self.dudx5q_tf))
        
        self.loss2 = 10 * tf.reduce_mean(tf.square(self.diva + self.f_tf))
        
        self.loss3 = 10 * tf.reduce_mean(
            tf.square(self.abdr[:, 0:1] - self.g1_tf) + tf.square(self.abdr[:, 1:2] - self.g2_tf) + tf.square(
                self.abdr[:, 2:3] - self.g3_tf) + tf.square(self.abdr[:, 3:4] - self.g4_tf) + tf.square(
                self.abdr[:, 4:5] - self.g5_tf))
                    
        #self.loss4 = tf.reduce_mean(tf.square(self.sig))
        
        self.loss5 = tf.reduce_mean(
            tf.square(self.sigx1) + tf.square(self.sigx2) + tf.square(self.sigx3) + tf.square(self.sigx4) + tf.square(
                self.sigx5))
        
        self.loss6 = 0.00001 * (self.loss5)
        
        self.loss7 = 1 * tf.reduce_mean(tf.square(self.sigb - self.sigbdr_tf))


        self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss6
        
        
        # initialize empty vct to store the loss values after each epoch for visualizing the loss
        self.loss_vals = []
        self.loss_vals1 = []
        self.loss_vals2 = []
        self.loss_vals3 = []
        self.loss_vals6 = []
        self.loss_vals7 = []
        
        
        
        # optimizer
        global_step = tf.Variable(0, trainable=False)
        start_learning_rate = 0.003
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 3000, 0.75)

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
    
    
    

    def net_a(self, x1, x2, x3, x4, x5):
        a = self.neural_net(tf.concat([x1, x2, x3, x4, x5], 1), self.weights2, self.biases2)
        return a

    def net_sigma(self, x1, x2, x3, x4, x5):
        sig = self.neural_net(tf.concat([x1, x2, x3, x4, x5], 1), self.weights1, self.biases1)
        gradsigx1 = tf.gradients(sig, x1)[0]
        gradsigx2 = tf.gradients(sig, x2)[0]
        gradsigx3 = tf.gradients(sig, x3)[0]
        gradsigx4 = tf.gradients(sig, x4)[0]
        gradsigx5 = tf.gradients(sig, x5)[0]

        return sig, gradsigx1, gradsigx2, gradsigx3, gradsigx4, gradsigx5

    def div1(self, x1, x2, x3, x4, x5):
        a = self.net_a(x1, x2, x3, x4, x5)
        ax1 = a[:, 0:1]
        ax2 = a[:, 1:2]
        ax3 = a[:, 2:3]
        ax4 = a[:, 3:4]
        ax5 = a[:, 4:5]

        gradax1 = tf.gradients(ax1, x1)[0]
        gradax2 = tf.gradients(ax2, x2)[0]
        gradax3 = tf.gradients(ax3, x3)[0]
        gradax4 = tf.gradients(ax4, x4)[0]
        gradax5 = tf.gradients(ax5, x5)[0]

        div = gradax1 + gradax2 + gradax3 + gradax4 + gradax5
        return div

    


    def train(self):

        tf_dict = {self.x1q_tf: self.x1q, self.x2q_tf: self.x2q, self.x3q_tf: self.x3q,
                   self.x4q_tf: self.x4q, self.x5q_tf: self.x5q,
                   self.x1b_tf: self.x1b, self.x2b_tf: self.x2b, self.x3b_tf: self.x3b,
                   self.x4b_tf: self.x4b, self.x5b_tf: self.x5b,
                   self.f_tf: self.f, self.sigbdr_tf: self.sigbdr,
                   self.g1_tf: self.g1, self.g2_tf: self.g2, self.g3_tf: self.g3,
                   self.g4_tf: self.g4, self.g5_tf: self.g5,
                   self.dudx1q_tf: self.dudx1q, self.dudx2q_tf: self.dudx2q, self.dudx3q_tf: self.dudx3q,
                   self.dudx4q_tf: self.dudx4q, self.dudx5q_tf: self.dudx5q}
        
        
        #########################################################
        ####################### IMPORTANT #######################
        #########################################################
        
        # check readme.md section 2 for saving and restoring initializations of weights and biases
        
        #saver = tf.train.Saver(tf.trainable_variables())
        #modeltoRestore = os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE.ckpt')
        #saver.restore(self.sess, modeltoRestore)

        for epoch in range(60000):
            self.sess.run(self.optimizer, feed_dict=tf_dict)
            
            iter_loss1 = self.sess.run(self.loss1, feed_dict=tf_dict)
            iter_loss2 = self.sess.run(self.loss2, feed_dict=tf_dict)
            iter_loss3 = self.sess.run(self.loss3, feed_dict=tf_dict)
            iter_loss6 = self.sess.run(self.loss6, feed_dict=tf_dict)
            iter_loss7 = self.sess.run(self.loss7, feed_dict=tf_dict)
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
                print("\t\tloss7 = " + str(1 * iter_loss7))
                print("Iter " + str(epoch) + ", s1 = " + str(s1) + ", s0 = " + str(s0))

            self.loss_vals.append(iter_loss)
            self.loss_vals1.append(1 * iter_loss1)
            self.loss_vals2.append(1 * iter_loss2)
            self.loss_vals3.append(1 * iter_loss3)
            self.loss_vals6.append(1 * iter_loss6)
            self.loss_vals7.append(1 * iter_loss7)
            
        #save_path = saver.save(self.sess, os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE.ckpt'))



    def predict(self, X_star):
        sigma = self.sess.run(self.sig,
                              {self.x1q_tf: X_star[:, 0:1], self.x2q_tf: X_star[:, 1:2], self.x3q_tf: X_star[:, 2:3],
                               self.x4q_tf: X_star[:, 3:4], self.x5q_tf: X_star[:, 4:5]})
        return sigma


if __name__ == "__main__":
    
    # huber constant for smoothing the square root function, used only in ex5.2 & ex5.8
    theta = 0.000001

    # feed forward NN architecture
    layers1 = [5, 26, 26, 26, 10, 1]
    layers2 = [5, 26, 26, 26, 10, 5]
    

    # load data
    data = scipy.io.loadmat('YOUR_PATH/ntdiridim5prj.mat')
    X_quad = data['X_quad']
    X_bdr = data['X_bdr']
    X_star = data['X_pred']
    f = data['f']
    sigbdr = data['sigbdr']
    g1 = data['g1']
    g2 = data['g2']
    g3 = data['g3']
    g4 = data['g4']
    g5 = data['g5']
    dudx1q = data['gux1quad']
    dudx2q = data['gux2quad']
    dudx3q = data['gux3quad']
    dudx4q = data['gux4quad']
    dudx5q = data['gux5quad']
    sigma_eval1 = data['sigma_eval1']


    # model input
    model = kohnint(X_quad, X_bdr, layers1, layers2, theta, f, g1, g2, g3, g4, g5,
                    dudx1q, dudx2q, dudx3q, dudx4q, dudx5q, sigbdr)
    model.X_star = X_star


    # train
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

 
    # predict
    sigpred = model.predict(X_star)

    # L2-relative error
    error_sigma = np.linalg.norm(sigpred - sigma_eval1, 2) / np.linalg.norm(sigma_eval1, 2)
    print('Error sigma: %e' % (error_sigma))
    
    # save the conductivity values at prediction points as .mat file
    scipy.io.savemat('YOUR_OUTPUT_FILE_NAME.mat', {'sigma': sigpred})

    

    # plot the loss
    plt.figure(1)
    x1 = np.arange(1, len(model.loss_vals) + 1)
    plt.semilogy(x1, model.loss_vals, label='l')
    plt.semilogy(x1, model.loss_vals1, label='1')
    plt.semilogy(x1, model.loss_vals2, label='2')
    plt.semilogy(x1, model.loss_vals3, label='3')
    plt.semilogy(x1, model.loss_vals6, label='6')
    plt.semilogy(x1, model.loss_vals7, label='7')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.ylim(1e-7, 1000)
    plt.legend(loc=0, ncol=2)
    plt.show()
