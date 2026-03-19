import os
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ode_bank import *
from model import CAE
from utils_ode import *

from datetime import datetime


'''CONFIG'''
TIME_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
# META_INFO = 'lorenz_chen_lv'      # Add more information here
META_INFO = 'lorenz_chen_lv_ori'      # Add more information here
TIME_STAMP = TIME_NOW + META_INFO
MODEL_NAME = f'cae_{TIME_STAMP}'
SAVE_DIR = f'./results/{MODEL_NAME}/'
# os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG_OVERWRITE = False
CAE_OVERWRITE = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = {
    # ode hyper parameter
    "grid_shape": [48, 48, 48],
    "grid_limits": [[-25,25], [-25,25], [0,50]],
    # "grid_limits": [[0,1], [0,1], [0,1]],    # Add more information here
    "input_dimension": 3,
    # data sampler hyper parameter
    "batch_size": 12,
    "depth": 64,
    "turb_scale": 0.1,
    # cae hyper parameter 
    "kernel_size": [5, 3, 3],
    "strides": [3, 2, 2],
    # pre training hyper parameter
    "cae_max_batches": 5000,
    "cae_num_logs": 10,
    "cae_learning_rate": 1e-3,
    "cae_reduce_factor": 0.8,
    "cae_reduce_patience": 100,
    "cae_early_stopping_rate": 1e-4,
    # transfer hyper parameter
    "transfer_max_iters": 100000,
    "transfer_num_logs": 1000,
    "transfer_learning_rate": 0.1,
    "transfer_reduce_factor": 0.8,
    "transfer_reduce_patience": 1000,
    "transfer_early_stopping_rate": 1e-6
}
# with open(SAVE_DIR + f'config_{MODEL_NAME}.json', 'w', newline='') as f:
#     json.dump(config, f, indent=4)
#     print(json.dumps(config, indent=4))


'''DATA SET'''
# _, hadley_unit_nv = ensemble_rescaling(hadley)
# _, genesio_tesi_unit_nv = ensemble_rescaling(genesio_tesi)
# _, halvorsen_unit_nv = ensemble_rescaling(halvorsen)
# _, rossler_unit_nv = ensemble_rescaling(rossler)
_, lorenz_unit_nv = ensemble_rescaling(lorenz)
_, chen_unit_nv = ensemble_rescaling(chen)
_, lv_unit_nv = ensemble_rescaling(lv)
# _, nose_hoover_unit_nv = ensemble_rescaling(nose_hoover)
# _, rucklidge_unit_nv = ensemble_rescaling(rucklidge)
# ode_train_list = [lorenz_unit_nv, chen_unit_nv, lv_unit_nv]        # Add more information here
ode_train_list = [lorenz, chen, lv]        # Add more information here
print([ode.name for ode in ode_train_list])                        # Add more information here


'''MODEL AND DATA SAMPLER'''
cae = CAE(cae_depth=config['depth'], cae_input_dimension=config['input_dimension'], 
          cae_strides=config['strides'], cae_kernel_size=config['kernel_size'])

data_sampler = DataSampler(grid_shape=config['grid_shape'], grid_limits=config['grid_limits'], 
                           turb_scale=config['turb_scale'])


class Fusion(object):
    def __init__(self,
                 cae,
                 data_sampler,
                 ):
        self.cae = cae.to(DEVICE)

        self.data_sampler = data_sampler

        self.mse_loss = nn.MSELoss()

    def pre_train(self, save_dir=None, save_name=None):
        if save_dir is None:
            save_dir = f'./checkpoints/'
        if save_name is None:
            save_name = f'{MODEL_NAME}.pt'

        save_file = save_dir + save_name
        os.makedirs(save_dir, exist_ok=True)

        if not CAE_OVERWRITE and os.path.exists(save_file):
            print('Loading CAE weights from checkpoint...')
            self.cae.load_state_dict(torch.load(save_file))
            print(f'CAE weights loaded from checkpoint! {save_file}')
            self.cae.eval()
            return None
        else:
            print('Start Training CAE...')

            self.cae_learning_rate = config['cae_learning_rate']

            self.cae_optimizer = optim.Adam(self.cae.parameters(), lr=self.cae_learning_rate)
            self.cae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.cae_optimizer, mode='min', 
                                    factor=config['cae_reduce_factor'], patience=config['cae_reduce_patience'])

            self.cae.train()

            info = []
            batches = []
            for i_batch in range(config['cae_max_batches']):
                images_tensor, _ = self.data_sampler.get_batch(batch_size=config['batch_size'], turb=True)
                images_tensor = images_tensor.to(DEVICE)
                self.cae_optimizer.zero_grad()

                _, _, images_rep_tensor = self.cae(images_tensor)

                mse_loss = self.mse_loss(images_rep_tensor, images_tensor)

                mse_loss.backward()
                self.cae_optimizer.step()
                self.cae_scheduler.step(mse_loss)
                current_lr = self.cae_scheduler.get_last_lr()[0]

                if not (i_batch+1) % config['cae_num_logs']:
                    print(f'Iter {i_batch+1:3d}, MSE Loss: {mse_loss:6.4e}, Learning Rate: {current_lr:6.4e}')
                    info.append(mse_loss.item())
                    batches.append(i_batch)
                    self.plot_info(info, batches=batches, save_npy=False)

                if current_lr < config['cae_early_stopping_rate']:
                    print(f'Early Stopping at iteration {i_batch + 1}')
                    break

            print('CAE Training Finished! Saving checkpoint...')
            torch.save(self.cae.state_dict(), save_file)
            print(f'CAE weights saved to checkpoint! {save_file}')
            self.cae.eval()

            return info

    def get_batch(self, batch_size, turb=True):
        images_tensor, names = self.data_sampler.get_batch(batch_size=batch_size, turb=turb)
        return images_tensor.to(DEVICE), names

    def plot_info(self, info, batches=None, xlabel='num_batches', save_dir=None, file_name_suffix='', save_npy=True):
        if save_dir is None:
            save_dir = f'./results/{MODEL_NAME}/'
        os.makedirs(save_dir, exist_ok=True)

        if not isinstance(info, np.ndarray):
            info = np.array(info)

        _ = plt.figure(figsize=(9.6, 7.2), dpi=300)
        if batches is None:
            _ = plt.plot(info)
        else:
            _ = plt.plot(batches, info)
        _ = plt.legend(['mse_loss'])
        _ = plt.xlabel(xlabel)
        _ = plt.savefig(save_dir + f'cae_info_{file_name_suffix}.png')
        plt.close()

        if save_npy:
            np.save(save_dir + f'cae_info_{file_name_suffix}', info)

    def gram_matrix(self, x):
        if len(x.shape) == 4:
            c, d, h, w = x.size()
            b = 1
        elif len(x.shape) == 5:
            b, c, d, h, w = x.size()
        else:
            raise ValueError(f'Input tensor shape {x.shape} not supported!')
        
        target_features = x.view(b, c, d * h * w)  # flatten the feature maps
        gram_product = torch.bmm(target_features, target_features.transpose(1, 2))  # compute the gram product
        return gram_product.div(c * d * h * w)

    def styl_loss(self, styl_features, fusion_features):
        return F.mse_loss(self.gram_matrix(styl_features), self.gram_matrix(fusion_features))

    def cont_loss(self, cont_features, fusion_features):
        return F.mse_loss(cont_features, fusion_features)

    def transfer(self, target_tensor_list, weight_list, init_tensor=None, save_dir=None, verbose=True):
        if save_dir is None:
            save_dir = f'./results/{MODEL_NAME}/'
        os.makedirs(save_dir, exist_ok=True)

        self.transfer_learning_rate = config['transfer_learning_rate']   

        if not isinstance(target_tensor_list, torch.Tensor):
            if len(target_tensor_list[0].shape) == 4:
                target_tensor_list = torch.stack(target_tensor_list, dim=0)
            elif len(target_tensor_list[0].shape) == 5:
                target_tensor_list = torch.cat(target_tensor_list, dim=0)
            else:
                raise ValueError(f'Input tensor shape {target_tensor_list[0].shape} not supported!') 
                          
        target_tensor_list = target_tensor_list.to('cuda')

        if init_tensor is not None:
            if len(init_tensor.shape) == 4:
                init_tensor = init_tensor.unsqueeze(0)
            fusion_tensor = init_tensor.clone().detach()
        else:
            fusion_tensor = torch.mean(target_tensor_list, dim=0, keepdim=True)
            fusion_tensor += torch.randn_like(fusion_tensor) * 0.01

        fusion_tensor = fusion_tensor.to('cuda').requires_grad_(True)

        self.transfer_optimizer = optim.Adam([fusion_tensor], lr=self.transfer_learning_rate)
        self.transfer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.transfer_optimizer, mode='min', 
                                    factor=config['transfer_reduce_factor'], patience=config['transfer_reduce_patience'])

        losses = []

        best_loss = float('inf')
        best_fusion_tensor = None

        for i_iter in range(config['transfer_max_iters']):
            self.transfer_optimizer.zero_grad()

            target_features_list, _, _ = self.cae(target_tensor_list)
            fusion_features, _, _ = self.cae(fusion_tensor)

            transfer_loss = 0.0
            loss_monitor = np.zeros((len(target_tensor_list), len(target_features_list), 2))
            for i_tensor in range(len(target_tensor_list)):
                for i_layer in range(len(target_features_list)):
                    target_features = target_features_list[i_layer][i_tensor]
                    cont_loss = self.cont_loss(target_features, fusion_features[i_layer][0])
                    styl_loss = self.styl_loss(target_features, fusion_features[i_layer][0])
                    
                    loss_monitor[i_tensor][i_layer][0] = cont_loss.item()
                    loss_monitor[i_tensor][i_layer][1] = styl_loss.item()

                    cont_weight = weight_list[i_tensor][0][i_layer]
                    if cont_weight:
                        transfer_loss += cont_weight * cont_loss

                    styl_weight = weight_list[i_tensor][1][i_layer]
                    if styl_weight:
                        transfer_loss += styl_weight * styl_loss
            
            current_loss = transfer_loss.item()

            if current_loss < best_loss:
                best_loss = current_loss
                best_fusion_tensor = fusion_tensor.detach().cpu().clone()

            losses.append(loss_monitor)

            transfer_loss.backward()
            self.transfer_optimizer.step()
            self.transfer_scheduler.step(transfer_loss)
            current_lr = self.transfer_scheduler.get_last_lr()[0]


            if verbose and not (i_iter + 1) % config['transfer_num_logs']:
                print(f"Iter {i_iter + 1}/{config['transfer_max_iters']}, Learning Rate: {current_lr:.6e}, Transfer Loss: {transfer_loss.item():.6e}, Best Loss: {best_loss:.6e}")


            if current_lr < config['transfer_early_stopping_rate']:
                if verbose:
                    print(f'Early Stopping at iteration {i_iter + 1}')
                break

        return best_fusion_tensor.detach().cpu(), np.array(losses)



if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)

    data_sampler.add_odes(ode_train_list)

    x, _ = data_sampler.get_batch(batch_size=12, turb=True)

    cae = cae.to(DEVICE)
 
    model = Fusion(cae, data_sampler)

    info = model.pre_train()

    # info: training mse loss
    if info is not None:
        model.plot_info(info, save_dir=SAVE_DIR)

    for ode in data_sampler.odes:
        ode.reset_coef()
