import torch, json, os
from model import CAE
from ode_bank import *
from utils_eval import show_tensor_slices, gram_matrix, show_gram
from utils_ode import pairwise_cae_distance_matrices, time_reverse_ode, permute_sign_conjugacy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_PATH = "./checkpoints/cae_20251203_163503lorenz_lv_rossler.pt"
META_INFO = 'lorenz_lv_rossler'

ROOT_PATH = './results_display/cae_results'
PIC_ROOT_PATH = ROOT_PATH + '/' + META_INFO
os.makedirs(PIC_ROOT_PATH, exist_ok=True)

MODEL_CONFIG = {
    "grid_shape" : [48, 48, 48],
    "grid_limits" : [[0,1], [0,1], [0,1]], 
    "turb_scale" : 0.0,
}

# PLOT_CONFIG = {
#     # original system
#     'plot original system': False,
#     # reconstructed system
# }

_, lorenz_unit_nv = ensemble_rescaling(lorenz)
# _, chen_unit_nv = ensemble_rescaling(chen)
_, lv_unit_nv = ensemble_rescaling(lv)
_, rossler_unit_nv = ensemble_rescaling(rossler)
# _, burke_shaw_unit_nv = ensemble_rescaling(burke_shaw)
ode_train_list = [lorenz_unit_nv, lv_unit_nv, rossler_unit_nv] 


cae = CAE().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
print("Loading checkpoint...")
state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
cae.load_state_dict(state_dict)
print('Done!')

cae.eval()

def cae_evaluator(system):
    with torch.no_grad():
        grids = calc_grids(grid_limits=MODEL_CONFIG["grid_limits"], grid_shape=MODEL_CONFIG["grid_shape"])
        vec = ode2vec(system, grids)
        print(vec.shape)
        vec = vec.transpose(3, 0, 1, 2)
        vec = torch.tensor(vec, dtype=torch.float32)
        vec = vec.unsqueeze(0)
        vec = vec.to(DEVICE, non_blocking=True, dtype=next(cae.parameters()).dtype)
        feats, z, vec_hat = cae(vec)
    
        reconstruction_mse = torch.mean((vec_hat - vec)**2, dim=(1,2,3,4))
        print("reconstruction MSE:", reconstruction_mse.cpu().numpy())

    vec_hat = vec_hat.squeeze(0)      
    vec_hat = vec_hat.cpu().numpy()
    vec_hat = vec_hat.transpose(1, 2, 3, 0)
    
    system_hat = vec2ode(
        vec=vec_hat, 
        grids=grids, 
        sparse_coef=None, 
        max_order=2, 
        extra_term_dicts=None,
        mode='linear', 
        alpha=0.1, 
        alphas=None, 
        cv=5, 
        fit_intercept=True,
        name=system.name + '_hat', 
        verbose=True, 
        return_diagnostics=False, 
        grid_limits=MODEL_CONFIG["grid_limits"]
    )

    system_hat.plot(pic_save_dir=PIC_ROOT_PATH+f'/{system_hat.name}_reconstructed.png', init_point=(0.1,0.1,0.1))


if __name__ == "__main__":
    cae_evaluator(lv_unit_nv)
    