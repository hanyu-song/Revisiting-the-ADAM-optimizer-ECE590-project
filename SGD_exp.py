from resnet20 import ResNetCIFAR
from train_util import train, test, test2
import torch
import numpy as np
#from prune import prune
import matplotlib.pyplot as pt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import pickle
from math import sqrt

method = 'SGD'
lr = 0.1
regs = np.linspace(-7, -3, num=5)
regs = 10 ** regs
epochs = 1800
batch_size = 128
# reg = reg_norm * \sqrt{b / B T}
ams = [False]
for amsgrad in ams:
    for reg in regs:
        print("reg = {}".format(reg))
        net = ResNetCIFAR(num_layers=20)
        net = net.to(device)
        loss, acc = train(net,opt_method= method, epochs = epochs, batch_size = batch_size, \
                  lr = lr, reg = reg, amsgrad = amsgrad, log_every_n=300)
        name = "summary_stats/net_{0}_{1}_amsgrad_{2}_epoch_{3}".format(method, int(reg * 1e7), amsgrad,epochs)
        open_file = open("{0}.p".format(name), "wb" )
        pickle.dump([acc, loss], open_file)
        open_file.close()
