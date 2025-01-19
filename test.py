import torch
import numpy as np


alphas = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/fractional_orders_alpha.npy')
print(alphas.shape)

inputs = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/fractional_system_inputs.npy')
print(inputs.shape)

x_init = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/fractional_system_trajectories.npy')
print(x_init.shape)

LQR_Q = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/LQR_Q.npy')
print(LQR_Q.shape)

LQR_R = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/LQR_R.npy')
print(LQR_R.shape)

optimal_controls = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_1000/optimal_control_U.npy')
print(optimal_controls)

A = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/system_matrix_A.npy')
print(A.shape)

B = np.load('/home/zhangpeiyu/llm/interpretability_llm/CFNO/data/data_10000filter/system_matrix_B.npy')
print(B.shape)

