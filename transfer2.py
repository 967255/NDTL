import torch, json, os
from model import CAE
from ode_bank import *
from pre_train import Fusion
from utils_ode import vec2tensor, tensor2vec

import time
from datetime import datetime


'''CONFIG'''
TIME_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
META_INFO = 'lorenz_chen_lv_ori'       
TIME_STAMP = TIME_NOW + META_INFO   
MODEL_NAME = f'fusion_{TIME_STAMP}'
SAVE_DIR = f'./results/{MODEL_NAME}/'
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT_PATH = "./checkpoints/cae_20260225_205314lorenz_chen_lv_ori.pt"  
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = {
    # ode hyper parameter
    "grid_shape": [48, 48, 48],
    "grid_limits": [[-25,25], [-25,25], [0,50]],
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
    "transfer_max_iters": 10000,
    "transfer_num_logs": 100,
    "transfer_learning_rate": 0.1,
    "transfer_reduce_factor": 0.5,
    "transfer_reduce_patience": 500,
    "transfer_early_stopping_rate": 1e-6
}

'''ODE TRAIN LIST'''
ode_train_list = [lorenz, chen, lv] 
print([ode.name for ode in ode_train_list])


'''MODEL AND DATA SAMPLER'''
cae = CAE(cae_depth=config['depth'], cae_input_dimension=config['input_dimension'], 
          cae_strides=config['strides'], cae_kernel_size=config['kernel_size'])

data_sampler = DataSampler(grid_shape=config['grid_shape'], grid_limits=config['grid_limits'], 
                           turb_scale=config['turb_scale'])

data_sampler.add_odes(ode_train_list)
print('Data sampler has been built!')

cae = cae.to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
print("Loading checkpoint...")
state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
cae.load_state_dict(state_dict)
cae.eval()
print('Cae has been loaded!')

model = Fusion(cae, data_sampler)
print('Fusion has prepared!')


def fusion_verify(sys_a: int, 
                  sys_b: int, 
                  style_strength: float = 0.01, 
                  cont_ratio_list: list = None, 
                  style_ratio_list: list = None
                  ):
    # build optimazation target
    target_ode_list = [ode_train_list[sys_a], ode_train_list[sys_b]]
    target_tensor_list = [vec2tensor(ode2vec(ode, data_sampler.grids)) for ode in target_ode_list]

    experiment_stamp = target_ode_list[0].name + '+' + target_ode_list[1].name
    func_save_dir = SAVE_DIR + f'/{experiment_stamp}/'
    os.makedirs(func_save_dir, exist_ok=True)

    L = 3 
    C = 1.0           # content total intensity
    S = style_strength        # treat style_strength as style total intensity


    grid_shape = tuple(data_sampler.grid_shape)
    init_image = np.random.normal(size=(*grid_shape, 3)).astype(np.float32)
    # init_image = np.random.normal(size=(*model.data_sampler.grid_shape, 3))
    init_tensor = vec2tensor(init_image)


    cont_ratio_list = cont_ratio_list if cont_ratio_list is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    styl_ratio_list = style_ratio_list if style_ratio_list is not None else [0.0, 0.5, 1.0]
    if np.isclose(S, 0.0):
        styl_ratio_list = [0.0]


    t0 = time.time()
    vecs = np.zeros((len(cont_ratio_list), len(styl_ratio_list), *grid_shape, 3), dtype=init_image.dtype)
    

    def build_weight_list(r_c, r_s, C, S, L):
        w_cont_a = [C * r_c / L] * L
        w_cont_b = [C * (1.0 - r_c) / L] * L

        w_styl_a = [S * r_s / L] * L
        w_styl_b = [S * (1.0 - r_s) / L] * L
        return [[w_cont_a, w_styl_a], [w_cont_b, w_styl_b]]


    for i_c, r_c in enumerate(cont_ratio_list):
        for i_s, r_s in enumerate(styl_ratio_list):
            log_file_name = f'rc{r_c:.2f}_rs{r_s:.2f}_C{C:.2f}_S{S:.2f}.log'
            with open(func_save_dir + log_file_name, 'a') as log_file:
                log_file.write(f'ratios: r_c={r_c:.2f}, r_s={r_s:.2f}; strengths: C={C:.2f}, S={S:.2f}\n')

            print(f'r_c={r_c:.2f}, r_s={r_s:.2f}, time={time.time() - t0:.2f}s')

            if np.isclose(C, 0.0) and np.isclose(S, 0.0):
                vecs[i_c, i_s] = init_image
                continue

            weight_list = build_weight_list(r_c, r_s, C, S, L)

            fusion_tensor, losses = model.transfer(
                target_tensor_list, weight_list,
                init_tensor=init_tensor, save_dir=func_save_dir, verbose=True
            )
            vec = tensor2vec(fusion_tensor)
            vecs[i_c, i_s] = vec

    np.save(SAVE_DIR + f'vecs_{target_ode_list[0].name}_{target_ode_list[1].name}_{MODEL_NAME}.npy', vecs)
    print('Done!')


if __name__ == '__main__':
    fusion_verify(sys_a=0,
                  sys_b=2,
                  style_strength=0.0,
                  cont_ratio_list=[0.25, 0.5, 0.75]
                  )                                