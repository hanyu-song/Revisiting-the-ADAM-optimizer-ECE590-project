from resnet20 import ResNetCIFAR
from train_util import train, test, test2
import torch
import numpy as np
import matplotlib.pyplot as pt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import pickle
from math import sqrt
import os

method = 'AdamW'
lr = 0.001
reg_norms = np.linspace(-6, -4.5, num=5)
reg_norms = 10 ** (reg_norms + 4)
epochs = 2
tot_train_size = 50000
batch_size = 128
# reg = reg_norm * \sqrt{b / B T}
ams = [False, True]
i=int(os.environ['SLURM_ARRAY_TASK_ID'])
for amsgrad in ams:
    reg_norm = reg_norms[i-1]
    reg = reg_norm * sqrt(batch_size / epochs / tot_train_size)
    print("reg = {}".format(reg))
    net = ResNetCIFAR(num_layers=20)
    net = net.to(device)
    loss, acc = train(net,opt_method= method, epochs = epochs, batch_size = batch_size, \
                  lr = lr, reg = reg, amsgrad = amsgrad, log_every_n=300)
    name = "summary_stats/net_{0}_{1}_amsgrad_{2}_epoch_{3}".format(method, int(reg_norm * 1e7), amsgrad,epochs)
    open_file = open("{0}.p".format(name), "wb" )
    pickle.dump([acc, loss], open_file)
    open_file.close()
