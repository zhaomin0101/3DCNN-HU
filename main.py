# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:33:46 2021

@author: zhao min

Zhao M, Wang M, Chen J, et al. Hyperspectral Unmixing for Additive Nonlinear Models With a 3-D-CNN Autoencoder Network[J].
IEEE Transactions on Geoscience and Remote Sensing, 2021.
"""
import os
import scipy.io as sio
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable

from models.multiscale_model_jas import autoencoder_model
# from pretrain_weight_encoder import pretrain_encoder
# from pretrain_weight_decoder import pretrain_decoder

from utils.ProgressBar import ProgressBar

model_name = 'autoencoder'
workspace = '/home/mzhao/A_deep_unmixing/AE_TGRS_NN'
torch.cuda.set_device(1)
GPU_NUMS = 1
EPOCH = 200
BATCH_SIZE = 400
learning_rate = 1e-3
num_endmember = 3
num_band = 156
la = 0.05
ga = 0.008


## --------------------- Load the data ---------------------------------------
N = 95*95;
hsi_name = 'data_samson_patch_3'
file_path = os.path.join(workspace, "data", "%s.mat" % hsi_name)
datamat = sio.loadmat(file_path)
hsi = datamat[hsi_name]
hsi = torch.from_numpy(hsi)
hsi = hsi[0:N, :, :]

endmember_name = 'weight_samson_long'
file_path = os.path.join(workspace, "data", "%s.mat" % endmember_name)
datamat = sio.loadmat(file_path)
W_init = datamat[endmember_name]
W_init = torch.from_numpy(W_init)

model = autoencoder_model()
model.decoder_linear[0].weight.data = W_init


model = model.cuda() if GPU_NUMS > 0 else model

print(model)

criterion = MSELoss()

## ----------------------------------------------------------------
if model_name == 'autoencoder':
    ignored_params = list(map(id, model.decoder_linear[0].parameters()))  # 需要微调的参数
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())  # 需要调整的参数
    optimizer = Adam([
        {'params': base_params},
        {'params': model.decoder_linear[0].parameters(), 'lr': 1e-4}
    ], lr=learning_rate, weight_decay=1e-5)

vector_all = []
linear_all = []
code_onehot = torch.eye(num_endmember)
code_onehot = Variable(code_onehot).cuda()

W_init = Variable(W_init).cuda()
data_loader = DataLoader(hsi, batch_size=BATCH_SIZE, shuffle=False)

proBar = ProgressBar(EPOCH, len(data_loader), "Loss:%.5f")

for epoch in range(1, EPOCH):
    l_item = 0
    for data in data_loader:
        pixel = data
        pixel_1 = Variable(pixel).cuda() if GPU_NUMS > 0 else Variable(pixel)
        # ===================forward=====================
        pixel = torch.reshape(pixel_1, (-1, 1, 3, 3, num_band))
        output_linear, out_nonlinear, vector = model(pixel)
        pixel = pixel_1[:,4,:]
        pixel = torch.reshape(pixel, (-1, num_band))

        output_linear_get = output_linear[:, 0:num_band] + output_linear[:, num_band:num_band*2] + output_linear[:, num_band*2:num_band*3]

        loss_reconstruction = criterion(output_linear_get+out_nonlinear, pixel)


        #l2
        l2_temp1 = model.decoder_nonlinear[0].weight
        l2_temp1 = l2_temp1.reshape(num_band, num_band * 3)
        l2_temp2 = model.decoder_nonlinear[2].weight
        l2_temp3 = model.decoder_nonlinear[4].weight
        l2_regularization = torch.cat((l2_temp1, l2_temp2, l2_temp3), 1)
        l2_regularization = torch.norm(l2_regularization)

        #smooth
        weight_temp = model.get_endmember(code_onehot)
        loss_diff_temp = weight_temp[:, 2:-1] - weight_temp[:, 1:-2]
        loss_diff_temp = loss_diff_temp.abs()
        loss_difference = loss_diff_temp.mean()

        loss = loss_reconstruction + la*l2_regularization + ga*loss_difference

                # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proBar.show(epoch, loss.item())
        # ===================log========================
        if epoch == EPOCH-1:
            vector_temp = vector.cpu().data
            vector_temp = vector_temp.numpy()
            vector_all = np.append(vector_all,vector_temp)
            vector_all = vector_all.reshape(-1, num_endmember)
        name_vector_all = 'vector_all_samson_pre.mat'
        sio.savemat(name_vector_all, {'vector_all': vector_all})

torch.save(model.encoder_cnn.state_dict(), 'samson_model.pth')
endmember = model.get_endmember(code_onehot)
endmember = endmember.cpu().data
endmember = endmember.numpy()
sio.savemat('endmember_pre_samson.mat', {'endmember': endmember})