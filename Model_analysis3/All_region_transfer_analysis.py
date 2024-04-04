import pandas as pd
import numpy as np
import json
import geopandas as gpd
import matplotlib as mpl
import copy
import draw_utils
import math
import utils
from copy import copy
import matplotlib.pyplot as plt
# state_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_state.shp', encoding = 'gb18030')
# tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding='gb18030')
# tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding='gb18030')
cbsa_name_list_all = ['NY','GLA','WB','SFB', 'GB','DV','AL', 'MM', 'PS',  'MSP']
p_all = []
for iii in range(10):
    #训练测试数据
    region = cbsa_name_list_all[iii]
    with open('./dataset/trans_data/out_flow_level_all_x_'+str(region)+'.json', 'r', encoding='UTF-8') as f:
        out_flow_level_all_x_train = json.load(fp=f)
    with open('./dataset/trans_data/out_flow_level_all_y_'+str(region)+'.json', 'r', encoding='UTF-8') as f:
        out_flow_level_all_y_train = json.load(fp=f)
    #建立估计模型
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    def linear_function(X, a, b):
        x0, x1 = X
        return (x0**a)/(x1**b)

    data_x = np.array(out_flow_level_all_x_train[0])
    data_y = np.array(out_flow_level_all_y_train[0]).reshape(-1,1)
    for kk in range(1,5):
        data_x = np.vstack((data_x,np.array(out_flow_level_all_x_train[kk])))
        data_y = np.vstack((data_y,np.array(out_flow_level_all_y_train[kk]).reshape(-1,1)))
    data_y = data_y.reshape(-1)

    params, covariance = curve_fit(linear_function, np.vstack((data_x[:,0], data_x[:,1])), data_y)
    print(params)
    pp = []
    for region_ind in cbsa_name_list_all:
        with open('./dataset/trans_data/out_flow_level_all_x_' + str(region_ind) + '_test.json', 'r', encoding='UTF-8') as f:
            out_flow_level_all_x_test = json.load(fp=f)
        with open('./dataset/trans_data/out_flow_level_all_y_' + str(region_ind) + '_test.json', 'r', encoding='UTF-8') as f:
            out_flow_level_all_y_test = json.load(fp=f)
        pre_result = []
        real_flow = []
        for leve in range(5):
            for bb in range(len(out_flow_level_all_x_test[leve])): #ind
                pre_x, pre_y = out_flow_level_all_x_test[leve][bb], out_flow_level_all_y_test[leve][bb] #ind bb
                pre_ = []
                for i in range(len(pre_x)):
                    pre_.append(pre_x[i][0]**params[0]/pre_x[i][1]**params[1])
                real_ = []
                for i in range(len(pre_)):
                    pre_[i] = round(pre_[i])
                    real_.append(round(pre_y[i]))
                pre_list = np.array(pre_)
                real_list = np.array(real_)
                pre_sum = np.sum(pre_list.flatten())
                real_sum = np.sum(real_list.flatten())
                if pre_sum == 0:
                    pre_flow_of_xi = pre_list.flatten()
                else:
                    pre_flow_of_xi = pre_list.flatten() / pre_sum * real_sum
                for i in range(len(pre_flow_of_xi)):
                    pre_flow_of_xi[i] = round(pre_flow_of_xi[i])
                pre_result.append(pre_flow_of_xi.reshape(1, -1))
                real_flow.append(real_list.reshape(1, -1))

        cpc = utils.cal_cpc_value2(pre_result, real_flow)
        rmse, pcc, mae = utils.cal_rmse2(pre_result, real_flow)
        # print(cpc)
        # print(rmse)
        # print(pcc)
        # print(mae)
        pp.append(cpc); pp.append(rmse); pp.append(pcc); pp.append(mae);
    p_all.append(pp)
p_all = np.array(p_all).T