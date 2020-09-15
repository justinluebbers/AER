
###
# Manage experiment settings such as which models to use
###
def get_model_names(env, perfect_models = []):
    if env == 'MountainCar':
        em_name = 'EM_MountainCar_n50k_s0.0_h4x32_d0.1_lr0.001_l-nrmse_range.ckpt'
        rm_name = 'RM_MountainCar_n15k_s0.0_h0x2_d0.0_lr0.001_l-nrmse_range.ckpt'
        tm_name = 'TM_MountainCar_n15k_s0.3_h2x8_d0.1_lr0.001_l-termination_loss.ckpt'
        num_timesteps = 300000
    elif env == 'CartPole':
        em_name = 'EM_CartPole_n50k_s0.0_h4x128_d0.1_lr0.001_l-nrmse_range.ckpt'
        rm_name = 'RM_CartPole_n15k_s0.0_h0x2_d0.0_lr0.001_l-nrmse_range.ckpt'
        tm_name = 'TM_CartPole_n20k_s0.3_h8x32_d0.1_lr0.001_l-termination_loss.ckpt'
        num_timesteps = 300000
    elif env == 'Acrobot':
        em_name = 'EM_Acrobot_n50k_s0.0_h4x128_d0.1_lr0.001_l-nrmse_range.ckpt'
        rm_name = 'RM_Acrobot_n20k_s0.0_h8x32_d0.0_lr0.001_l-nrmse_range.ckpt'
        tm_name = 'TM_Acrobot_n20k_s0.3_h4x128_d0.1_lr0.001_l-termination_loss.ckpt'
        num_timesteps = 300000
    else:
        em_name = ''
        rm_name = ''
        tm_name = ''
        num_timesteps = 0
    for m in perfect_models:
        if m == 'RM':
            rm_name = rm_name + '.perfect'
        elif m == 'TM':
            tm_name = tm_name + '.perfect'
        elif m == 'EM':
            em_name = em_name + '.perfect'
            
    return em_name, rm_name, tm_name, num_timesteps
