# ECE590_finalproject

A. Introduction of python and data set files

a. resnet20.py and pruned_layers.py contain the implementation of ResNet-20 (source: ECE590 Homework 3).

b. train_util contains training an architecture on the CIFAR10 data set using either SGD, Adam, or AdamW (with or without AMSGrad) algorithm.

c. adam.py contains my implementation of Adam and AdamW algorithm (with or without AMSGrad).
(code reference: https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py).

d. adam_exp2.py, adamW_exp2.py and SGD_exp2.py contains the code of the experiments for evaluating their performance on the CIFAR10 data set. If you woud like to run them on a computing cluster with GPUs, adam_exp.py, adamW_exp.py and SGD_exp.py are the python scripts for running the Slurm job arrays, and Adam.sh, AdamW.sh and SGD.sh are the bash scripts for Slurm.

e. makePlots.ipynb is the notebook for reproducing figures in the report/poster using the saved summary statistics (e.g. training loss) in the directory of summary_stats

f. In the directory of data, there is the CIFAR10 data set.




B. Introduction of saved files 

a. (directory) check_points: check points of the trained ResNet-20 models on the CIFAR10 data set. The names are in the format of "net_{method}_{weight decay parameter * 1e7}_amsgrad_{amsgrad status}_epoch_{total number of epochs}.pt". In particular, total number of epochs = 1800 and method = SGD, Adam or AdamW.

b. (directory) summary_stats: summary statistics from the entire run, with the names being in the format of  "net_{method}_{weight decay parameter * 1e7}_amsgrad_{amsgrad status}_epoch_{total number of epochs}.p". In particular, total number of epochs = 1800 and method = SGD, Adam or AdamW. There are two variables saved in each file, with the first variable being accuracies and the second one being losses. Both variables are a numpy array with dimensions (number of epochs * 2), with the first column associated with the training set and second column associated with the test set. For example, accuracy[5,0] refers to the training accuracy at Epoch 5; loss[100, 1] refers to the test loss at Epoch 100.

