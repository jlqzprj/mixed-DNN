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
    def __init__(self, X_quad, X_full, X_bdr, layers1, layers2, theta, f, g1, g2, dudxq, dudyq, sigbdr):
        
        self.theta = theta
        
        self.xq = X_quad[:, 0:1]
        self.yq = X_quad[:, 1:2]
        
        self.xf = X_full[:, 0:1]
        self.yf = X_full[:, 1:2]
        
        self.sigbdr = sigbdr
        
        self.xb = X_bdr[:, 0:1]
        self.yb = X_bdr[:, 1:2]
        
        self.f = f
        self.g1 = g1 #(g1,g2) = q^dagger*\grad(z^delta) in the paper, corresponds to the dirichlet condition
        self.g2 = g2

        self.layers1 = layers1
        self.layers2 = layers2
        
        self.dudxq = dudxq
        self.dudyq = dudyq


        # Initialize trainable parameters weights and biases
        self.weights1, self.biases1 = self.initialize_NN(self.layers1)
        self.weights2, self.biases2 = self.initialize_NN(self.layers2)

        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))


        # placeholders were created to ease batch training
        self.xq_tf = tf.placeholder(tf.float32, shape=[None, self.xq.shape[1]])
        self.yq_tf = tf.placeholder(tf.float32, shape=[None, self.yq.shape[1]])
        self.xf_tf = tf.placeholder(tf.float32, shape=[None, self.xf.shape[1]])
        self.yf_tf = tf.placeholder(tf.float32, shape=[None, self.yf.shape[1]])
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.yb_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        
        self.g1_tf = tf.placeholder(tf.float32, shape=[None, self.g1.shape[1]])
        self.g2_tf = tf.placeholder(tf.float32, shape=[None, self.g2.shape[1]])
        self.f_tf = tf.placeholder(tf.float32, shape=[None, self.f.shape[1]])
        self.dudxq_tf = tf.placeholder(tf.float32, shape=[None, self.dudxq.shape[1]])
        self.dudyq_tf = tf.placeholder(tf.float32, shape=[None, self.dudyq.shape[1]])
        self.sigbdr_tf = tf.placeholder(tf.float32, shape=[None, self.sigbdr.shape[1]])
        
        
        
        # net_a here is the network for the vector field sigma in the paper
        # xq,yq are the collocation points sampled from the subdomain w
        self.a = self.net_a(self.xq_tf, self.yq_tf)
        # net_sigma here is the network for the conductivity q in the paper
        self.sig, self.sigx, self.sigy = self.net_sigma(self.xq_tf, self.yq_tf)
        # xf,yf are the collocation points sampled from the whole domain Omega
        self.sigf, self.sigxf, self.sigyf = self.net_sigma(self.xf_tf, self.yf_tf)
        self.sigb,_,_ = self.net_sigma(self.xb_tf, self.yb_tf)
        self.diva = self.div1(self.xf_tf, self.yf_tf)
        self.ab = self.net_a(self.xb_tf, self.yb_tf)
        
        
        
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



        # loss
        self.loss1 = 0.64 * tf.reduce_mean(tf.square(self.a[:, 0:1] - self.sigprj * self.dudxq_tf) + tf.square(
            self.a[:, 1:2] - self.sigprj * self.dudyq_tf))
        
        self.loss2 = 10 * tf.reduce_mean(tf.square(self.diva + self.f_tf))
        
        #self.loss3 = tf.reduce_mean(tf.square(self.sig))
        self.loss4 = tf.reduce_mean(tf.square(self.sigxf) + tf.square(self.sigyf))
        
        self.loss5 = 0.00001 * (self.loss4)
        
        self.loss6 = 1 * tf.reduce_mean(tf.square(self.sigb-self.sigbdr_tf))
        
        self.loss7 = 40 * tf.reduce_mean(tf.square(self.ab[:,0:1]-self.g1_tf)+tf.square(self.ab[:,1:2]-self.g2_tf))
        

        self.loss = self.loss1 + self.loss2 + self.loss5 + self.loss7
        
        
        
        # initialize empty vct to store the loss values after each epoch for visualizing the loss
        self.loss_vals = []
        self.loss_vals1 = []
        self.loss_vals2 = []
        self.loss_vals5 = []
        self.loss_vals6 = []
        self.loss_vals7 = []

        
        # optimizer
        global_step = tf.Variable(0, trainable=False)
        start_learning_rate = 0.003
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 2500, 0.8)

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




    def net_a(self, x, y):
        a = self.neural_net(tf.concat([x, y], 1), self.weights2, self.biases2)
        return a

    def net_sigma(self, x, y):
        sig = self.neural_net(tf.concat([x, y], 1), self.weights1, self.biases1)
        gradsigx = tf.gradients(sig, x)[0]
        gradsigy = tf.gradients(sig, y)[0]
        return sig, gradsigx, gradsigy

    def div1(self, x, y):
        a = self.net_a(x, y)
        ax = a[:, 0:1]
        ay = a[:, 1:2]
        gradax = tf.gradients(ax, x)[0]
        graday = tf.gradients(ay, y)[0]
        div = gradax + graday
        return div

    


    def train(self):

        tf_dict = {self.xq_tf: self.xq, self.yq_tf: self.yq, self.xb_tf: self.xb,
                   self.yb_tf: self.yb, self.f_tf: self.f, self.g1_tf: self.g1, self.g2_tf: self.g2,
                   self.dudxq_tf: self.dudxq,
                   self.dudyq_tf: self.dudyq, self.sigbdr_tf: self.sigbdr,self.xf_tf: self.xf, self.yf_tf: self.yf}
         
        
        #########################################################
        ####################### IMPORTANT #######################
        #########################################################
        
        # check readme.md section 2 for saving and restoring initializations of weights and biases
        
        
        #saver = tf.train.Saver(tf.trainable_variables())
        #modeltoRestore = os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE.ckpt')
        #saver.restore(self.sess, modeltoRestore)
        

        for epoch in range(80000):
            self.sess.run(self.optimizer, feed_dict=tf_dict)
            
            iter_loss1 = self.sess.run(self.loss1, feed_dict=tf_dict)
            iter_loss2 = self.sess.run(self.loss2, feed_dict=tf_dict)
            iter_loss5 = self.sess.run(self.loss5, feed_dict=tf_dict)
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
                print("\t\tloss5 = " + str(1 * iter_loss5))
                print("\t\tloss6 = " + str(1 * iter_loss6))
                print("\t\tloss7 = " + str(1 * iter_loss7))
                print("Iter " + str(epoch) + ", s1 = " + str(s1) + ", s0 = " + str(s0))
                
            self.loss_vals.append(iter_loss)
            self.loss_vals1.append(1 * iter_loss1)
            self.loss_vals2.append(1 * iter_loss2)
            self.loss_vals5.append(1 * iter_loss5)
            self.loss_vals6.append(1 * iter_loss6)
            self.loss_vals7.append(1 * iter_loss7)
            
        #save_path = saver.save(self.sess, os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE.ckpt'))


    def predict(self, X_star):
        sigma = self.sess.run(self.sigf, {self.xf_tf: X_star[:, 0:1], self.yf_tf: X_star[:, 1:2]})
        return sigma



if __name__ == "__main__":
    
    # huber constant for smoothing the square root function, used only in ex5.2 & ex5.8
    theta = 0.000001

    # feed forward NN architecture
    layers1 = [2, 26, 26, 26, 10, 1]
    layers2 = [2, 26, 26, 26, 10, 2]
    
    # load data
    data = scipy.io.loadmat('YOUR_PATH/diripartial2d.mat')
    X_quad = data['X_quad']
    X_full = data['X_quadfull']
    X_bdr = data['X_bdr']
    X_star = data['X_pred']
    f = data['f']
    g1 = data['g1']
    g2 = data['g2']
    sigbdr = data['sigbdr']
    dudxq = data['guxquad']
    dudyq = data['guyquad']
    sigma_eval1 = data['sigma_eval1']


    # model input
    model = kohnint(X_quad, X_full, X_bdr, layers1, layers2, theta, f, g1, g2, dudxq, dudyq, sigbdr)
    model.X_star = X_star
    model.sigma_eval1 = sigma_eval1


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
    plt.semilogy(x1, model.loss_vals,label='_nolegend_')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.ylim(1e-4, 10)
    plt.show()

