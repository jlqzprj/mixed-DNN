# mixed-DNN
Python codes for reproducing experiements in the paper https://arxiv.org/abs/2303.16454.


1. Visualization:

   The prediction/approximation of the conductivity q at selected prediction points is saved in 'YOUR_OUTPUT_FILE_NAME.mat' by the command line:
   
   'scipy.io.savemat('YOUR_OUTPUT_FILE_NAME.mat', {'sigma': sigpred})'
   
   at the end of the code. 


   The plots (exact q, learned q, and pointwise error) shown in the paper can be generated by running 'visualization.m' (with suitable changes, e.g., your own path/file name) in Matlab.



2. 
