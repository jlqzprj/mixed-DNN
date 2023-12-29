# mixed-DNN
Python codes for reproducing experiements in the paper https://arxiv.org/abs/2303.16454.


1. Visualization:

   The prediction / approximation of the conductivity q at selected prediction points is saved in 'YOUR_OUTPUT_FILE_NAME.mat' by the command line:
   
   'scipy.io.savemat('YOUR_OUTPUT_FILE_NAME.mat', {'sigma': sigpred})'
   
   at the end of the code. 


   The plots (exact q, learned q, and pointwise error) shown in the paper can be generated by running 'visualization.m' (with suitable changes, e.g., your own path / file name) in Matlab.




2. Initialization:

   The loss function 'self.loss' is designed according to formulations (3.6 & 3.7) and (4.4 & 4.5) in the paper.

   The projection operator P_A requires good initialization of the network q_theta (i.e., c_0 <= q_theta <= c_1), otherwise poor initializations of q_theta will be projected onto either c_0 or c_1, leading to suboptimal steps in the optimization process.

   To ensure the network q_theta is initialized well, one can either

   (a) Try out different initialization methods for the weights and biases, until the initialization satisfies c_0 <= q_theta <= c_1;

   or

   (b)

   (i) Run the script without the projection operator (i.e., replace 'self.sigprj' with 'self.sig' in 'self.loss') for a few epochs (200 in our case) until q_theta satisfies the inequality c_0 <= q_theta <= c_1.

      Uncomment the lines

      'saver = tf.train.Saver(tf.trainable_variables())'

      and

      'saver.save(self.sess, os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE_NAME.ckpt'))'

      in the train section to save values of the trainable variables (i.e., weights and biases).


   (ii) Run the script again with the projection operator (i.e., self.sigprj) and the variables stored in step (i) as the initialization of the weights and biases.

      The latter is achieved by uncommenting the line

      'saver.restore(self.sess, os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE_NAME.ckpt'))'

      and commenting the line
   
      'saver.save(self.sess, os.path.join('YOUR_PATH', 'YOUR_VARIABLE_STORE_FILE_NAME.ckpt'))'

      in the train section.


   (iii) Remark: if by default setting, your kernel does not reset all graphs each time you rerun the script, then you need to add an additional line

      'tf.reset_default_graph()'

      at the beginning of the code before implementing step (ii), to avoid naming problems of the variables.



3. References

   [1] Tensorflow: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/all_symbols

   [2] Tensorflow: https://github.com/tensorflow/docs/tree/master/site/en/r1

   [3] Feed forward architecture: https://github.com/maziarraissi/PINNs







