# 作者:王勇
# 开发时间:2024/5/4 18:31
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import callbacks
from scipy import io as sio
import matplotlib.pyplot as plt
import sys
import warnings
import scipy.linalg as splin
warnings.filterwarnings("ignore")
import numpy as np
import os
import h5py
import scipy as sp

def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos>1.0: cos = 1.0
    return np.arccos(cos)

def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    #for i in range(num_endmembers):
        #endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        #endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1
    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, ASAM / float(num)


def plotEndmembersAndGT_gray(endmembers, endmembersGT, wavelength_range=(0, 200)):
    """
    绘制端元和真值端元的光谱曲线为灰度图，增加分辨度
    """

    if not os.path.exists("./samson_endmembers_gray"):
        os.makedirs("./samson_endmembers_gray")

    num_endmembers, num_bands = endmembers.shape
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    # 重新匹配端元顺序
    hat, sad = order_endmembers(endmembersGT, endmembers)
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_bands)

    # 对端元进行归一化
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    # 绘图
    for i in range(num_endmembers):
        plt.figure(figsize=(8, 6))

        # 绘制预测端元（虚线）
        plt.plot(wavelengths, endmembers[hat[i], :], linestyle='--', color='red', linewidth=2.0, label='OUR')

        # 绘制真值端元（实线）
        plt.plot(wavelengths, endmembersGT[i, :], linestyle='-', color='blue', linewidth=2.0, label='SSJA')

        # 设置坐标轴和网格
        plt.ylim((0.2, 1))
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xlabel('Band', fontsize=15)
        plt.ylabel('Reflectance', fontsize=15)
        plt.title(f'Endmember {i + 1}', fontsize=16)
        plt.grid( linestyle=':', linewidth=0.5)

        # 添加图例
        plt.legend(fontsize=12, loc='upper right')

        # 保存灰度图
        save_path = os.path.join("./sy_endmembers", f'endmember_{i}.svg')
        plt.savefig(save_path, format='svg', dpi=300)
        plt.show()
        plt.close()


data = sio.loadmat("./Results/0_SSABN/Sy20_data/Sy20_data_run1.mat")
data_t = sio.loadmat("./result_p/DAEU_SSABN/sy_data/sy_data_run1.mat")
A = data['A']     #main A,DH E_est,TA E
B = data_t['A']
# B= np.transpose(B, (1, 0))    #if DH,运行
plotEndmembersAndGT_gray(A,B)
