import pandas as pd
import numpy as np
import json
import utils
from scipy.optimize import curve_fit
cbsa_name_list_all = ['NY','GLA','WB','SFB', 'GB','DV','AL', 'MM', 'PS',  'MSP']
def linear_function(X, a, b):
    x0, x1 = X
    return (x0 ** a) / (x1 ** b)

#%% 训练测试集准备
ensemble_train_data_x, ensemble_train_data_y = [],[]
ensemble_test_data_x, ensemble_test_data_y = [],[]

for source in range(10): #10个区域，每个区域的每个水平的数据分别拟合用于预测其他10个区域的数据。
    #训练测试数据
    region = cbsa_name_list_all[source]
    with open('./dataset/trans_data/out_flow_level_all_x_'+str(region)+'.json', 'r', encoding='UTF-8') as f:
        out_flow_level_all_x_train = json.load(fp=f)
    with open('./dataset/trans_data/out_flow_level_all_y_'+str(region)+'.json', 'r', encoding='UTF-8') as f:
        out_flow_level_all_y_train = json.load(fp=f)

    params_set= []
    for level_num in range(5):
        data_x_ind = np.array(out_flow_level_all_x_train[level_num])
        data_y_ind = np.array(out_flow_level_all_y_train[level_num])
        params_ind, covariance = curve_fit(linear_function, np.vstack((data_x_ind[:,0], data_x_ind[:,1])), data_y_ind)
        params_set.append(params_ind)
    # params_set_array = np.array(params_set)
    # params_set_array = pd.DataFrame(params_set_array)
    # params_set_array.to_csv('./dataset/sepr_data/' + str(region) + '_gravity_params_5levels.csv')
    to_data_x,to_data_y = [],[]
    for target in range(10):
        #导入其它区域的不同level的训练数据
        region = cbsa_name_list_all[target]
        with open('./dataset/trans_data/out_flow_level_all_x_' + str(region) + '.json', 'r', encoding='UTF-8') as f:
            out_flow_level_all_x_train_ind = json.load(fp=f)
        with open('./dataset/trans_data/out_flow_level_all_y_' + str(region) + '.json', 'r', encoding='UTF-8') as f:
            out_flow_level_all_y_train_ind = json.load(fp=f)

        #构建集成学习模型的训练集
        pre_result = [] #在目标区域 source 训练的模型在 target 区域测试的结果。5*len
        real_flow = []
        for leve in range(5):
            pre_x_all,pre_y_all = [],[]
            for (pre_x,pre_y) in zip(out_flow_level_all_x_train_ind[leve],out_flow_level_all_y_train_ind[leve]):  # ind
                pre_x_all.append(round(pre_x[0] ** params_set[leve][0] / pre_x[1] ** params_set[leve][1]))
                pre_y_all.append(round(pre_y))
            pre_result.append(pre_x_all)
            real_flow.append(pre_y_all)
        to_data_x.append(pre_result)
        to_data_y.append(real_flow)
    ensemble_train_data_x.append(to_data_x)
    ensemble_train_data_y.append(to_data_y)

    #构造测试集的集成训练结果
    to_data_x, to_data_y = [], []
    for target in range(10):
        # 导入其它区域的不同level的训练数据
        region = cbsa_name_list_all[target]
        with open('./dataset/trans_data/out_flow_level_all_x_' + str(region) + '_test.json', 'r', encoding='UTF-8') as f:
            out_flow_level_all_x_test_ind = json.load(fp=f)
        with open('./dataset/trans_data/out_flow_level_all_y_' + str(region) + '_test.json', 'r', encoding='UTF-8') as f:
            out_flow_level_all_y_test_ind = json.load(fp=f)
        # 构建集成学习模型的训练集
        pre_result = []  # 在目标区域 source 训练的模型在 target 区域测试的结果。5*len
        real_flow = []
        for leve in range(5):
            pre_x_all, pre_y_all = [], []
            for (pre_xs, pre_ys) in zip(out_flow_level_all_x_test_ind[leve], out_flow_level_all_y_test_ind[leve]):  # ind
                pre_x_all_ind,pre_y_all_ind = [],[]
                for (pre_x, pre_y) in zip(pre_xs, pre_ys):
                    pre_x_all_ind.append(round(pre_x[0] ** params_set[leve][0] / pre_x[1] ** params_set[leve][1]))
                    pre_y_all_ind.append(round(pre_y))
                pre_x_all.append(pre_x_all_ind)
                pre_y_all.append(pre_y_all_ind)
            pre_result.append(pre_x_all)
            real_flow.append(pre_y_all)
        to_data_x.append(pre_result)
        to_data_y.append(real_flow)
    ensemble_test_data_x.append(to_data_x)
    ensemble_test_data_y.append(to_data_y)

#%%调整数据结构，按水平来划分：[5, len, 10]
train_data_x,train_data_y = [],[]
test_data_x,test_data_y = [],[]
for level_num in range(5): #按水平来存储
    train_data_x.append([])
    train_data_y.append([])
    test_data_x.append([])
    test_data_y.append([])

for level in range(5): #5
    for region_d in range(10): #10 把同一区域的预测结果拼起来
        indx, indy = [], []
        for region_o in range(10): #10
            indx1,indy1 = [],[]
            for i in range(len(ensemble_train_data_x[region_o][region_d][level])):
                indx1.append(ensemble_train_data_x[region_o][region_d][level][i])
                indy1.append(ensemble_train_data_y[region_o][region_d][level][i])
            indx.append(indx1)
            indy.append(indy1)
        indx = np.array(indx).T; indy = np.array(indy).T #[len,10]
        train_data_x[level].append(indx)
        train_data_y[level].append(indy)

for level in range(5): #5
    for region_d in range(10): #10 把同一区域的预测结果拼起来
        indx, indy = [], []
        for o_num in range(len(ensemble_test_data_x[region_o][region_d][level])):  # 177 o个数
            indx1, indy1 = [], []
            for region_o in range(10): #10
                indx1.append(ensemble_test_data_x[region_o][region_d][level][o_num])
                indy1.append(ensemble_test_data_y[region_o][region_d][level][o_num])
            indx1 = np.array(indx1).T; indy1 = np.array(indy1).T #3*10
            indx.append(indx1)  #3*10
            indy.append(indy1)
        test_data_x[level].append(indx)
        test_data_y[level].append(indy)

#%% 模型训练+测试
p_all = []
for region_id in range(10): #目的地区域id
    region_name = cbsa_name_list_all[region_id]
    pre_result = []
    real_flow = []
    for level in range(5):
        train_data_x_level0 = train_data_x[level][region_id]
        train_data_y_level0 = train_data_y[level][region_id]
        test_data_x_level0 = test_data_x[level][region_id]
        test_data_y_level0 = test_data_y[level][region_id]

        from xgboost import XGBRegressor
        xgb_reg = XGBRegressor(n_estimators=100, max_depth=3)
        xgb_reg.fit(train_data_x_level0, train_data_y_level0[:,0]) #不同目的地区域的xgb可能不一样，因为目的地的训练数据不同

        # pickle.dump(xgb_reg, open("./dataset/sepr_data/xgb_reg_d="+str(region_name)+"_level="+str(level)+".dat", "wb"))

        for index in range(len(test_data_x_level0)): #ind
            pre_x, pre_y = test_data_x_level0[index], test_data_y_level0[index] #ind bb
            pre_x = np.array(pre_x)
            pre_ = xgb_reg.predict(pre_x)
            real_ = []
            for i in range(len(pre_)):
                pre_[i] = round(pre_[i])
                real_.append(round(pre_y[i][0]))
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
    print(cpc)
    print(rmse)
    print(pcc)
    print(mae)
    pp=[]
    pp.append(cpc); pp.append(rmse); pp.append(pcc); pp.append(mae);
    p_all.append(pp)
    # 1\5--200    3--2000   4\8--400  6--300   7--100  9\0-2-500
    ori_num = 0
    if region_id==0 or region_id==2 or region_id==9:
        des_num = 500
    elif region_id==1 or region_id==5:
        des_num =200
    elif region_id==3:
        des_num =2000
    elif region_id==4 or region_id==8:
        des_num =400
    elif region_id==6:
        des_num =300
    elif region_id==7:
        des_num =100
    import Visualization_prediction_scatter_bin_function
    Visualization_prediction_scatter_bin_function.draw_scatter_bin(region_name, "GM-level_CLVI_ensemble model", 10, real_flow, pre_result, ori_num, des_num)

p_all = np.array(p_all).T