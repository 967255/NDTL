import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils_ode import *
from ode import *



'''configuration'''
TIME_STAMP = 'lv_seris_cae'

MODEL_NAME = f'transfer_{TIME_STAMP}'
SAVE_DIR = f'./results/{TIME_STAMP}/'

os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG_OVERWRITE = False
CAE_OVERWRITE = False
IMAGE_OVERWRITE = True

if CONFIG_OVERWRITE or not os.path.exists(SAVE_DIR + f'config_{MODEL_NAME}.json'):
    print('Using config dict...')
    config = {
        "grid_limits": [[0, 1], [0, 1], [0, 1]], # the range of the vector field after scaling
        # "grid_limits": [[-25, 25], [-25, 25], [0, 50]],
        # "grid_limits_origin": [[-25, 25], [-25, 25], [0, 50]],
        "grid_shape": [48, 48, 48], # shape of input vector field is (48, 48, 48, 3)?  
        "input_dimension": 3, # shape of input vector field is (48, 48, 48, 3)? 

        "batch_size": 12,
        "depth": 64, # ？
        "turb_scale": 0.1,

        "kernel_size": [5, 3, 3], # 5 means channel is 5?
        "strides": [3, 2, 2], # ?

        # pre training hyper parameter
        "cae_max_batches": 5000,
        "cae_num_logs": 10,
        "cae_learning_rate": 1e-3,
        "cae_reduce_factor": 0.8,
        "cae_reduce_patience": 100,
        "cae_early_stopping_rate": 1e-6,
        
        # transfer hyper parameter
        "transfer_max_iters": 100000,
        "transfer_num_logs": 1000,
        "transfer_learning_rate": 0.1,
        "transfer_reduce_factor": 0.8,
        "transfer_reduce_patience": 1000,
        "transfer_early_stopping_rate": 1e-6
    }
    with open(SAVE_DIR + f'config_{MODEL_NAME}.json', 'w', newline='') as f:
        json.dump(config, f, indent=4)
        print(json.dumps(config, indent=4))
else:
    print('Loading config from JSON file...')
    config = json.load(open(SAVE_DIR + f'config_{MODEL_NAME}.json'))


'''model and functions'''
activation = nn.ReLU()


class DataSampler(object):
    def __init__(self):
        self.odes = []

        self.grid_shape = config['grid_shape']
        self.grids = calc_grids(grid_limits=config['grid_limits'], grid_shape=config['grid_shape'])

        self.max_order = 0 
        self.dimension = 0
        self.extra_term_dicts = {}
        
        self.X = calc_X(self.grids)

    def get_batch(self, batch_size, indexes=None, turb=True, turb_scale=None):
        vecs = []
        if indexes is None:
            indexes = np.random.randint(len(self.odes), size=(batch_size, ))
        elif isinstance(indexes, (int, float)):
            indexes = np.array([indexes]) 

        if turb_scale is None:
            turb_scale = config['turb_scale']

        for index in indexes:
            if turb:
                self.odes[index].random_coef()
            else:
                self.odes[index].reset_coef()

            vec = ode2vec(ode=self.odes[index], grids=self.grids)
            if turb:
                vec += turb_scale * np.random.randn(*vec.shape)

            vecs.append(vec)
        
        self.odes[index].reset_coef()
        vecs = np.array(vecs)
        # x.shape = (batch_size, 3, D, H, W)
        vecs = vecs.transpose(0, 4, 1, 2, 3)  # 转换轴顺序以匹配 (batch_size, channels, D, H, W)
        return torch.tensor(vecs, dtype=torch.float32), [self.odes[index].name for index in indexes]

    def add_odes(self, odes):
        for ode in odes:
            self.odes.append(ode)
            if self.max_order < ode.max_order:
                self.max_order = ode.max_order
            if self.dimension < ode.dimension:
                self.dimension = ode.dimension   
            self.extra_term_dicts.update(ode.extra_term_dicts)
            print(f'ODE {ode.name} added to Data Sampler!')
        
    def remove_all_odes(self):
        self.odes = []
        print('All ODEs removed')

## Create a dummy input to visualize the network structure
# dummy_input = torch.randn(1, 3, 48, 48, 48)  # (batch_size, channels, D, H, W)
# cae = CAE()
# feature_layers, z, x_ = cae(dummy_input)
## feature_layers: type=list, len=3, type(feature_layers[0]=torch.tensor)
## feature_layers[0]: torch.Size([1, 64, 16, 16, 16])
## feature_layers[1]: torch.Size([1, 128, 8, 8, 8])
## feature_layers[2]: torch.Size([1, 256, 4, 4, 4])
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.depth = config['depth']
        self.input_dimension = config['input_dimension']
        self.n_layers = len(config['strides'])
        self.layers = [int(2**i_layer * self.depth) for i_layer in range(self.n_layers)]
        self.strides = config['strides']
        if isinstance(config['kernel_size'], int):
            self.kernel_size = [config['kernel_size']] * self.n_layers
        else:
            self.kernel_size = config['kernel_size']
        
        self.encoder = nn.ModuleList()
        in_channels = self.input_dimension
        for i_layer in range(self.n_layers):
            self.encoder.append(
                nn.Conv3d(in_channels, self.layers[i_layer], kernel_size=self.kernel_size[i_layer], stride=self.strides[i_layer], padding=(self.kernel_size[i_layer]-1)//2)
            )
            in_channels = self.layers[i_layer]
        
        self.decoder = nn.ModuleList()
        for i_layer in range(self.n_layers-1, -1, -1):
            self.decoder.append(
                nn.ConvTranspose3d(self.layers[i_layer], self.layers[i_layer-1] if i_layer > 0 else self.input_dimension, kernel_size=self.kernel_size[i_layer], stride=self.strides[i_layer], padding=(self.kernel_size[i_layer]-1)//2, output_padding=self.strides[i_layer]-1)
            )
        
    def forward(self, x):      
        z = x.clone()
        # print(f'Input layer shape: {z.shape}')

        feature_layers = []
        # feature_layers.append(z)
        for i_layer, layer in enumerate(self.encoder):
            z = layer(z)
            feature_layers.append(z)
            # print(f'Encoder layer {i_layer} output shape: {z.shape}')
            z = activation(z)
        
        x_ = z.clone()
        for i_layer, layer in enumerate(self.decoder):
            x_ = layer(x_)
            if i_layer < len(self.decoder) - 1:  # 最后一层不需要激活
                x_ = activation(x_)
            # print(f'Decoder layer {i_layer} output shape: {x_.shape}')
        
        return feature_layers, z, x_


class Transfer(object):
    def __init__(self, cae, data_sampler):
        self.cae = cae.to('cuda')

        self.data_sampler = data_sampler

        self.mse_loss = nn.MSELoss()

    def pre_train(self, save_dir=None, save_name=None):
        if save_dir is None:
            save_dir = f'./checkpoints/'
        if save_name is None:
            save_name = f'{MODEL_NAME}.pt'

        save_file = save_dir + save_name
         
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
            self.cae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.cae_optimizer, mode='min', factor=config['cae_reduce_factor'], patience=config['cae_reduce_patience'])

            self.cae.train()

            info = []
            batches = []
            for i_batch in range(config['cae_max_batches']):
                images_tensor, _ = self.data_sampler.get_batch(batch_size=config['batch_size'], turb=True)
                images_tensor = images_tensor.to('cuda')
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
        return images_tensor.to('cuda'), names

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
            np.save(save_dir + f'cae_info_{file_name_suffix}.png', info)

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
            # fusion_tensor = torch.randn_like(target_tensor_list[0]).unsqueeze(0) * 0.01

        fusion_tensor = fusion_tensor.to('cuda').requires_grad_(True)

        self.transfer_optimizer = optim.Adam([fusion_tensor], lr=self.transfer_learning_rate)
        self.transfer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.transfer_optimizer, mode='min', factor=config['transfer_reduce_factor'], patience=config['transfer_reduce_patience'])

        losses = []
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
            
            losses.append(loss_monitor)

            transfer_loss.backward()
            self.transfer_optimizer.step()
            self.transfer_scheduler.step(transfer_loss)
            current_lr = self.transfer_scheduler.get_last_lr()[0]


            if verbose and not (i_iter + 1) % config['transfer_num_logs']:
                print(f"Iter {i_iter + 1}/{config['transfer_max_iters']}, Learning Rate: {current_lr:.6e}, Transfer Loss: {transfer_loss.item():.6e}")
                

            if current_lr < config['transfer_early_stopping_rate']:
                if verbose:
                    print(f'Early Stopping at iteration {i_iter + 1}')
                break

        return fusion_tensor.detach().cpu(), np.array(losses)


def tensor2vec(image_tensor):
    # from tensor(1, channels, D, H, W) to ndarray(D, H, W, channels)
    return image_tensor.squeeze(0).permute(1, 2, 3, 0).numpy()


# from ndarray(D, H, W, channels) to tensor(1, channels, D, H, W)
def vec2tensor(image_vec):
    # from ndarray(D, H, W, channels) to tensor(1, channels, D, H, W)
    return torch.tensor(image_vec, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)




'''pre-train and fine-tune'''
# [lorenz_shifted, chen_shifted, lv_shifted, seris, rossler_shifted, prs]
ode_train_list = [ode_shifted_list[2], ode_shifted_list[3]]
print([ode.name for ode in ode_train_list])


data_sampler = DataSampler()
data_sampler.add_odes(ode_train_list)
# x, n = data_sampler.get_batch(4)


cae = CAE().to('cuda')

model = Transfer(cae, data_sampler)

info = model.pre_train()

# info: training mse loss
if info is not None:
    model.plot_info(info, save_dir=SAVE_DIR)

for ode in data_sampler.odes:
    ode.reset_coef()




'''calculate r0, not that important'''
# input: a ode(object);
# output: r0 of the ode
def calc_r0(ode): 
    coef = ode.coef # coef ia a numpy.ndarray, take lorenz for example, lorenz.coef: shape=(3,10) cause variable=3 order=2;
    # 定义传染矩阵 T（传染过程相关的主要项）
    T = np.array([
        [0, coef[0, 2], coef[0, 3]],
        [0, 0, coef[1, 2]],
        [0, 0, 0]
    ])


    # 定义恢复矩阵 Sigma（恢复过程相关的主要项）
    Sigma = np.array([
        [-coef[0, 1], 0, 0],
        [coef[1, 1], -coef[1, 2], 0],
        [coef[2, 1], coef[2, 2], -coef[2, 2]]
    ])

    # 计算恢复矩阵的逆矩阵
    Sigma_inv = np.linalg.inv(Sigma)

    # 计算下一代矩阵 G
    G = T @ Sigma_inv

    # 计算 G 的特征值，并选取最大的特征值作为 R0
    eigenvalues = np.linalg.eigvals(G)
    r0 = max(eigenvalues).real
    return r0


config['transfer_max_iters'] = 200000
config['transfer_num_logs'] = 1000
config['transfer_reduce_factor'] = 0.5
config['transfer_reduce_patience'] = 500