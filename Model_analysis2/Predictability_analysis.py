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
state_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_state.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding='gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding='gb18030')
cbsa_name_list_all = ['NY','GLA','WB','SFB', 'GB','DV','AL', 'MM',  'PS', 'MSP']

#%% [基于bsa筛选区域，county、tract、group]
region = cbsa_name_list_all[9]
model_name = "gravity"
dataset_name = 'Weeplace_'+region
cbsa_set = pd.read_csv("./dataset/Region/"+str(region)+".csv", header=None)  # cbsa region set NY_NJ_PA_BSA
#tract tess
for i in range(len(cbsa_set)):
    if i==0:
        tess_tract_selected = tess_tract_all[(tess_tract_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_tract_all['COUNTYNAME']==cbsa_set.iloc[i][0])]
        current_len = len(tess_tract_selected)
    else:
        tess_tract_selected = pd.concat((tess_tract_selected,tess_tract_all[(tess_tract_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_tract_all['COUNTYNAME']==cbsa_set.iloc[i][0])]))
        if len(tess_tract_selected)<=current_len:
            print(1)
            print(i)
        current_len = len(tess_tract_selected)
#group tess
for i in range(len(cbsa_set)):
    if i==0:
        tess_group_selected = tess_all[(tess_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_all['COUNTYNAME']==cbsa_set.iloc[i][0])]
        current_len = len(tess_group_selected)
    else:
        tess_group_selected = pd.concat((tess_group_selected,tess_all[(tess_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_all['COUNTYNAME']==cbsa_set.iloc[i][0])]))
        if len(tess_group_selected) <= current_len:
            print(2)
            print(i)
        current_len = len(tess_group_selected)

tess_tract_income = []
tess_tract_pop = []
for i in range(len(tess_tract_selected)):
    geoid = tess_tract_selected.iloc[i].GEOID
    tess_ind = tess_group_selected[tess_group_selected['index'].str.startswith(geoid)]
    income_sum = tess_ind['INCOME'].sum()
    pop_sum = tess_ind['POPULATION'].sum()
    tess_tract_income.append(income_sum)
    tess_tract_pop.append(pop_sum)
tess_tract_selected['INCOME'] = tess_tract_income
tess_tract_selected['POPULATION'] = tess_tract_pop
tess_tract_selected = tess_tract_selected.reset_index(drop=True); tess_tract_selected = tess_tract_selected.to_crs("epsg:4269");tess_tract_selected.crs
tess_group_selected = tess_group_selected.reset_index(drop=True); tess_group_selected = tess_group_selected.to_crs("epsg:4269");tess_group_selected.crs

#%%  数据加载, 转化为od flow形式，并存储flow seg 和 loc name 至 value_flow_seg_name_list
user_in_MSA_final = {}
with open('./dataset/' + str(dataset_name) + '/user_travel_in_' + str(region) + '.json', 'r', encoding='UTF-8') as f:
    user_in_MSA_final = json.load(fp=f)
location_segregation_value = {}
with open('./dataset/' + str(dataset_name) + '/Location_segregation_value_in_region_' + str(region) + '1.json', 'r',
          encoding='UTF-8') as f:
    location_segregation_value = json.load(fp=f)
for key, v in location_segregation_value.items():
    location_segregation_value[key] = 1 - v
location_segregation_value_all = {}

with open('./dataset/' + str(dataset_name) + '/All_locations_segregation_value1.json', 'r', encoding='UTF-8') as f:
    location_segregation_value_all = json.load(fp=f)  # 部分location是不在目标区域内的
for key, v in location_segregation_value_all.items():
    location_segregation_value_all[key] = 1 - v  # 0表示不隔离,1表示隔离
user_home_info_all_final = {}
with open('./dataset/' + str(dataset_name) + '/user_home_info_all_final.json', 'r', encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f)

##%% 统计 od flow data
odflow_msa = {}
msa_set = {}
for key, value in user_in_MSA_final.items():
    for i in range(1, len(value)):
        o_county_id = str(value[i - 1][4]) + str(value[i - 1][5]) + str(value[i - 1][7])  # +str(value[i-1][8])
        d_county_id = str(value[i][4]) + str(value[i][5]) + str(value[i][7])  # +str(value[i][8])
        if (str(o_county_id), str(d_county_id)) not in odflow_msa.keys():
            odflow_msa[(str(o_county_id), str(d_county_id))] = 1
        else:
            odflow_msa[(str(o_county_id), str(d_county_id))] += 1
        msa_set[str(o_county_id)] = 1
        msa_set[str(d_county_id)] = 1

o2d2flow_msa = {}
for (o, d), f in odflow_msa.items():
    try:
        d2f = o2d2flow_msa[o]
        d2f[d] = f
    except KeyError:
        o2d2flow_msa[o] = {d: f}

o_flow_sum = {}
for key, values in o2d2flow_msa.items():
    ss = 0
    for subkey,va in values.items():
        ss+=va
    o_flow_sum[key] = ss

subkey_set,subkey_set_rev = {},{}
tess_tract_selected
for i in range(len(tess_tract_selected)):
    subkey_set[i] = []
    subkey_set[i].append(str(tess_tract_selected.iloc[i].GEOID))
    subkey_set[i].append(tess_tract_selected.iloc[i].POPULATION)
    subkey_set[i].append(tess_tract_selected.iloc[i].geometry.centroid.x)
    subkey_set[i].append(tess_tract_selected.iloc[i].geometry.centroid.y)
    subkey_set_rev[str(tess_tract_selected.iloc[i].GEOID)] = i

#%% select model 存储flow\seg\loc name value
value_flow_seg_name_list = []
for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():  # 出发点约束在我们的研究区域内 3077个loc 符合条件
        flow_all_ = 0
        for subkey, va in values.items():
            flow_all_ += va
        ind = []
        ind.append(location_segregation_value[key]);
        ind.append(len(values));
        ind.append(flow_all_);
        ind.append(key);
        value_flow_seg_name_list.append(ind)
    else:
        print(key)
value_flow_seg_name_list = sorted(value_flow_seg_name_list)  # 基于loc的seg values 进行升序排序

#  基于出行数据，筛选研究区域的tess
tess_selected_final = tess_tract_selected[tess_tract_selected['GEOID'] == tess_tract_selected.iloc[0].GEOID]
tess_ind = {}
tess_ind[tess_tract_selected.iloc[0].GEOID] = 1
for key, values in o2d2flow_msa.items():
    if key not in tess_ind.keys():
        tess_ind[key] = 1
        tess_selected_final = pd.concat((tess_selected_final, tess_tract_selected[tess_tract_selected['GEOID'] == key]))
    for sub_key, va in values.items():
        if sub_key not in tess_ind.keys():
            tess_ind[sub_key] = 1
            tess_selected_final = pd.concat((tess_selected_final, tess_tract_selected[tess_tract_selected['GEOID'] == sub_key]))
tess_selected_final = tess_selected_final.reset_index(drop=True);tess_selected_final = tess_selected_final.to_crs("epsg:4269");tess_selected_final.crs

#  排序存储flow和seg
value_flow_dict = {}
for i in range(len(value_flow_seg_name_list)):  # 将根据seg value进行出行flow 合并，计算seg相同的loc的总flow
    if value_flow_seg_name_list[i][0] not in value_flow_dict.keys():  # seg\degree\flow——all\o_id
        value_flow_dict[value_flow_seg_name_list[i][0]] = []
        value_flow_dict[value_flow_seg_name_list[i][0]].append(value_flow_seg_name_list[i][0])  # 0：seg values
        value_flow_dict[value_flow_seg_name_list[i][0]].append(value_flow_seg_name_list[i][3])  # 3：oid
    else:
        value_flow_dict[value_flow_seg_name_list[i][0]][0] += value_flow_seg_name_list[i][0]
        value_flow_dict[value_flow_seg_name_list[i][0]].append(value_flow_seg_name_list[i][3])

value_seg_list_sort = []  # 基于seg的累积隔离值进行升序
for key, values in value_flow_dict.items():
    value_seg_list_sort.append(values)
value_seg_list_sort = sorted(value_seg_list_sort)

# 基于洛伦兹曲线计算各个level的位置
ss_sum = 0.0
for i in range(len(value_seg_list_sort)):
    ss_sum += value_seg_list_sort[i][0]

value_flow_cir_level_1,value_rank_cir_level_1 = [],[]
current, ss, k = 0., 0, 0
for i in range(len(value_seg_list_sort)):
    k += 1
    value_flow_cir_level_1.append((ss + value_seg_list_sort[i][0]) / ss_sum)
    value_rank_cir_level_1.append(k / len(value_seg_list_sort))
    ss += value_seg_list_sort[i][0]

def draw(data1, data2, level_num):
    plot = plt.figure()
    ax = plot.add_subplot(1, 1, 1)
    ax.plot(data1, data1, linestyle='--', color='black')
    ax.plot(data1, data2, linewidth=1.5, color='blue', label="Seg values")
    plt.xlabel("Rank value")  # $r$
    plt.ylabel("Cumulative flow volume")  # P(Flow$_{i}$>$seg$)
    # plt.savefig("./Figure/Seg_value_of_different_seg_levels_"+str(level_num)+"_" + str(region) + ".png", dpi=360, bbox_inches='tight')
    # plt.show()

def calculate_slope(x, y):
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    slope = dy / dx
    # print(dy);
    # print(dx);
    y_intercept = y[-1] - slope * x[-1]
    x_intersection = -y_intercept / slope
    return slope, x_intersection

value_seg_cir_level_all, value_rank_cir_level_all, loc_level_all,x_intersection,slopes = [], [], [],[],[]
value_seg_cir_level_all.append(value_flow_cir_level_1); value_rank_cir_level_all.append(value_rank_cir_level_1)
inter_pre = 1
for ii in range(20):
    slope1, inter_new = calculate_slope(value_rank_cir_level_all[ii], value_seg_cir_level_all[ii])
    print(f"Slope at level {ii}: {slope1}")
    print(f"x_intersection: {inter_new}")
    x_intersection.append(inter_new)
    slopes.append(slope1)
    #统计剔除之前的区域后，还有多少个region，以及总的ss有多少
    ss_sum, k, tot = 0., 0., 0
    for i in range(len(value_seg_list_sort)):
        k += 1
        if k / len(value_seg_list_sort) <= inter_new: #剩余的更不隔离的地区的总个数和seg值
            ss_sum += value_seg_list_sort[i][0]; tot += 1

    ss, k = 0., 0.0
    value_seg_cir_level_ind, value_rank_cir_level_ind, loc_level_ind = [], [], []
    for i in range(len(value_seg_list_sort)):
        k += 1
        if k / len(value_seg_list_sort) <= inter_new: #剩余那些不隔离的
            value_seg_cir_level_ind.append((ss + value_seg_list_sort[i][0]) / ss_sum)
            value_rank_cir_level_ind.append(k / tot)
            ss += value_seg_list_sort[i][0]
        elif k / len(value_seg_list_sort) <= inter_pre:
            for j in range(1, len(value_seg_list_sort[i])):
                loc_level_ind.append(value_seg_list_sort[i][j])

    #update info
    inter_pre = inter_new
    if len(value_seg_cir_level_ind)>=len(value_seg_cir_level_all[ii]):
        ss, k, loc_level_ind = 0., 0.0, []
        for i in range(len(value_seg_list_sort)):
            k += 1
            if k / len(value_seg_list_sort) <= inter_new:  # 剔除那些非更不隔离的
                for j in range(1, len(value_seg_list_sort[i])):
                    loc_level_ind.append(value_seg_list_sort[i][j])
        loc_level_all.append(loc_level_ind)
        break
    value_seg_cir_level_all.append(value_seg_cir_level_ind)
    value_rank_cir_level_all.append(value_rank_cir_level_ind)
    loc_level_all.append(loc_level_ind)

def dd_all():
    width, color_num = 3, 12
    colors = np.linspace(0.2, 1.0, color_num)
    color_map = plt.get_cmap('Blues')  # Greens
    plot = plt.figure()
    ax = plot.add_subplot(1, 1, 1)
    ax.plot(value_rank_cir_level_all[0], value_rank_cir_level_all[0], linewidth=2, linestyle='--', color='black')
    for i in range(len(value_rank_cir_level_all)):
        ax.plot(value_rank_cir_level_all[i], value_seg_cir_level_all[i], linewidth=width, color=color_map((5-i)*2 / color_num),
                label="Seg level" +str(i+1))
    plt.xlabel("Rank")  # $r$
    plt.ylabel("Cumulative segregation values")  # P(Flow$_{i}$>$seg$)
    plt.legend()
    plt.show()
dd_all()

#%%  分析5个level之间的可预测性 #为每个o点分配到train or test中，根据每个level中，选20的数据作为test，放入train_list建一个list存储
max_level_num = 5
train_list,train_list_level,test_list,test_list_level,all_origion,origion_level = [],[],[],[],[],[]
abs_error = []
abs_error.append([])
for i in range(max_level_num):
    train_list_level.append([])
    test_list_level.append([])
    origion_level.append([])
    abs_error.append([])

for key,values in o_flow_sum.items():
    if key in location_segregation_value.keys():
        o_seg = location_segregation_value[key]
        ind = []; ind.append(values); ind.append(key)
        for i in range(max_level_num):
            if o_seg > x_intersection[i]:
                origion_level[i].append(ind)
                break
            if i == max_level_num - 1:
                origion_level[i].append(ind)

for i in range(max_level_num):
    origion_level_i = sorted(origion_level[i])
    for j in range(len(origion_level_i)):
        if j%5!=0:
            train_list_level[i].append(origion_level_i[j][1])
            train_list.append(origion_level_i[j][1])
        else:
            test_list_level[i].append(origion_level_i[j][1])
            test_list.append(origion_level_i[j][1])

#%% 构建训练测试数据
leve = 0
out_flow_level_all_x, out_flow_level_all_y, out_flow_all_x,out_flow_all_y = [],[],[],[]
out_flow_level_all_id, out_flow_all_id= [],[]
se_list_level_all, se_list_all = [], []
for i in range(max_level_num):
    out_flow_level_all_x.append([])
    out_flow_level_all_id.append([])
    out_flow_level_all_y.append([])

for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():
        seg_v = location_segregation_value[key]
        ind_all_x, ind_all_y, ind_all_id = [],[],[]
        ss = 0
        for subkey, va in o2d2flow_msa[key].items():
            ss+=va
        for subkey, va in o2d2flow_msa[key].items():
            if subkey==key:
                continue
            if subkey_set[subkey_set_rev[key]][1]<=100 or subkey_set[subkey_set_rev[subkey]][1]<=100:
                continue
            ind1,ind2 = [],[]
            ind1.append(subkey_set[subkey_set_rev[subkey]][1]) #d pop
            dis = utils.earth_distance([subkey_set[subkey_set_rev[key]][2],subkey_set[subkey_set_rev[key]][3]],
                                       [subkey_set[subkey_set_rev[subkey]][2],subkey_set[subkey_set_rev[subkey]][3]])
            ind1.append(dis)
            ind2.append(key); ind2.append(subkey)
            ind_all_x.append(ind1); ind_all_id.append(ind2)
            ind_all_y.append((va))

        if len(ind_all_y)>0:
            if key in train_list:
                out_flow_all_x.extend(ind_all_x)
                out_flow_all_y.extend(ind_all_y)
                out_flow_all_id.extend(ind_all_id)
            for i in range(max_level_num):
                if seg_v > x_intersection[i] and (key in train_list_level[i]):
                    for k in range(len(ind_all_x)):
                        out_flow_level_all_x[i].append(ind_all_x[k])
                        out_flow_level_all_y[i].append(ind_all_y[k])
                        out_flow_level_all_id[i].append(ind_all_id[k])
                    break
                if i == max_level_num-1 and (key in train_list_level[max_level_num-1]):
                    for k in range(len(ind_all_x)):
                        out_flow_level_all_x[i].append(ind_all_x[k])
                        out_flow_level_all_y[i].append(ind_all_y[k])
                        out_flow_level_all_id[i].append(ind_all_id[k])
                    break

#%%建立估计模型
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
def linear_function(X, a, b):
    x0, x1 = X
    return (x0**a)/(x1**b)

data_x = np.array(out_flow_all_x)
data_y = np.array(out_flow_all_y)
params, covariance = curve_fit(linear_function, np.vstack((data_x[:,0], data_x[:,1])), data_y)

data_x0 = np.array(out_flow_level_all_x[0]); data_y0 = np.array(out_flow_level_all_y[0])
data_x1 = np.array(out_flow_level_all_x[1]); data_y1 = np.array(out_flow_level_all_y[1])
data_x2 = np.array(out_flow_level_all_x[2]); data_y2 = np.array(out_flow_level_all_y[2])
data_x3 = np.array(out_flow_level_all_x[3]); data_y3 = np.array(out_flow_level_all_y[3])
data_x4 = np.array(out_flow_level_all_x[4]); data_y4 = np.array(out_flow_level_all_y[4])
params0, covariance = curve_fit(linear_function, np.vstack((data_x0[:,0], data_x0[:,1])), data_y0)
params1, covariance = curve_fit(linear_function, np.vstack((data_x1[:,0], data_x1[:,1])), data_y1)
params2, covariance = curve_fit(linear_function, np.vstack((data_x2[:,0], data_x2[:,1])), data_y2)
params3, covariance = curve_fit(linear_function, np.vstack((data_x3[:,0], data_x3[:,1])), data_y3)
params4, covariance = curve_fit(linear_function, np.vstack((data_x4[:,0], data_x4[:,1])), data_y4)

print(params0)
print(params1)
print(params2)
print(params3)
print(params4)

#%% test all
max_level_num = 5
out_flow_level_all_x, out_flow_level_all_y, out_flow_all_x,out_flow_all_y = [],[],[],[]
out_flow_level_all_id, out_flow_all_id= [],[]
se_list_level_all, se_list_all = [], []
for i in range(max_level_num):
    out_flow_level_all_x.append([])
    out_flow_level_all_id.append([])
    out_flow_level_all_y.append([])

for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():
        seg_v = location_segregation_value[key]
        ind_all_x, ind_all_y, ind_all_id = [],[],[]
        for subkey, va in o2d2flow_msa[key].items():
            # if subkey not in location_segregation_value.keys():
            #     continue
            if subkey==key:
                continue
            if subkey_set[subkey_set_rev[key]][1]<=100 or subkey_set[subkey_set_rev[subkey]][1]<=100:
                continue
            ind1,ind2 = [],[]
            ind1.append((subkey_set[subkey_set_rev[subkey]][1])) #d pop
            dis = utils.earth_distance([subkey_set[subkey_set_rev[key]][2],subkey_set[subkey_set_rev[key]][3]],
                                       [subkey_set[subkey_set_rev[subkey]][2],subkey_set[subkey_set_rev[subkey]][3]])
            ind1.append((dis))
            ind2.append(key); ind2.append(subkey)
            ind_all_x.append(ind1); ind_all_id.append(ind2)
            ind_all_y.append((va))
        if len(ind_all_y)>0:
            if key in test_list:
                out_flow_all_x.append(ind_all_x)
                out_flow_all_y.append(ind_all_y)
                out_flow_all_id.append(ind_all_id)
            for i in range(max_level_num):
                if seg_v > x_intersection[i] and (key in test_list_level[i]):
                    out_flow_level_all_x[i].append(ind_all_x)
                    out_flow_level_all_y[i].append(ind_all_y)
                    out_flow_level_all_id[i].append(ind_all_id)
                    break
                if i == max_level_num-1 and (key in test_list_level[max_level_num-1]):
                    out_flow_level_all_x[i].append(ind_all_x)
                    out_flow_level_all_y[i].append(ind_all_y)
                    out_flow_level_all_id[i].append(ind_all_id)

#%% 测试一起训练的总模型
pre_result = []
real_flow = []
seg_abs_error = []
for bb in range(len(out_flow_all_x)): #ind
    pre_x, pre_y,pre_id = out_flow_all_x[bb], out_flow_all_y[bb], out_flow_all_id[bb] #ind bb
    if location_segregation_value[pre_id[0][0]] > x_intersection[0]:
        params_ind = params0
    elif location_segregation_value[pre_id[0][0]] > x_intersection[1]:
        params_ind = params1
    elif location_segregation_value[pre_id[0][0]] > x_intersection[2]:
        params_ind = params2
    elif location_segregation_value[pre_id[0][0]] > x_intersection[3]:
        params_ind = params3
    else:
        params_ind = params4
    pre_ = []
    for i in range(len(pre_x)):
        pre_.append(pre_x[i][0]**params[0]/pre_x[i][1]**params[1])
        # pre_.append((pre_x[i][0]**params_ind[0])/(pre_x[i][1]**params_ind[1]))
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
    abs_e_sum = 0
    for i in range(len(pre_flow_of_xi)):
        abs_e_sum += abs(pre_flow_of_xi[i] - real_list[i])
        abs_error[0].append((pre_flow_of_xi[i]-real_list[i]))
    abs_e_sum /= len(pre_flow_of_xi)
    ind = []; ind.append(abs_e_sum); ind.append(location_segregation_value[pre_id[0][0]])
    seg_abs_error.append(ind)

    pre_result.append(pre_flow_of_xi.reshape(1, -1))
    real_flow.append(real_list.reshape(1, -1))

cpc = utils.cal_cpc_value2(pre_result, real_flow)
rmse, pcc, mae = utils.cal_rmse2(pre_result, real_flow)
print(cpc)
print(rmse)
print(pcc)
print(mae)

# seg_abs_error = np.array(seg_abs_error)
# seg_abs_error = pd.DataFrame(seg_abs_error)
# seg_abs_error.to_csv('./Figure/'+str(region)+'_seg-abs-error.csv')

def draw_scatter(data1, data2, x_name, y_name, fig_name):
    font1 = {'family': 'Arial', 'color': 'Black', 'size': 16}
    figsize = 6, 6
    plt.rcParams['xtick.direction'] = 'in'  ####坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in'  ####坐标轴刻度朝内
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='x', length=4, width=1, which='minor', top='on')  # ,top=True
    ax.tick_params(axis='x', length=8, width=1, which='major', top='on')  # ,right=True
    ax.tick_params(axis='y', length=8, width=1, which='major', right='on')  # ,top=True
    ax.tick_params(axis='y', length=4, width=1, which='minor', right='on')  # ,right=True
    ax.spines['bottom'].set_linewidth(1);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);  ####设置上部坐标轴的粗细
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    # ax.set_ylim(0, 30)
    # ax.set_xlim(0, 2)
    plt.scatter(data1,data2, cmap='jet', alpha=0.95, s=50, edgecolor='black')
    # import seaborn as sns
    # sns.kdeplot(data1, data2, cmap='Blues', fill=True)
    plt.xlabel(x_name, fontdict=font1)
    plt.ylabel(y_name, fontdict=font1)
    plt.xticks(fontsize=16)  # 横轴刻度字体大小
    plt.yticks(fontsize=16)  # 纵轴刻度字体大小
    plt.savefig("./Figure/" + str(fig_name) + ".png", dpi=360, bbox_inches='tight')
    plt.show()
# draw_scatter(seg_abs_error[:,0], seg_abs_error[:,1],   "abs_error", "seg", str(region)+'_seg-abs-error')
# plt.show()
# from scipy import stats
# pcc_flow_seg = stats.spearmanr(seg_abs_error[:,0], seg_abs_error[:,1])

#%% test level
for leve in range(5):
    if leve==0:
        pp_r = params0.copy()
    if leve==1:
        pp_r = params1.copy()
    if leve==2:
        pp_r = params2.copy()
    if leve==3:
        pp_r = params3.copy()
    if leve==4:
        pp_r = params4.copy()
    pre_result = []
    real_flow = []
    for bb in range(len(out_flow_level_all_x[leve])): #ind
        pre_x, pre_y = out_flow_level_all_x[leve][bb], out_flow_level_all_y[leve][bb] #ind bb
        pre_ = []
        for i in range(len(pre_x)):
            # pre_.append(pre_x[i][0]**params[0]/pre_x[i][1]**params[1])
            pre_.append(pre_x[i][0]**pp_r[0]/pre_x[i][1]**pp_r[1])
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
            abs_error[leve+1].append((pre_flow_of_xi[i]-real_list[i]))
        pre_result.append(pre_flow_of_xi.reshape(1, -1))
        real_flow.append(real_list.reshape(1, -1))

    cpc = utils.cal_cpc_value2(pre_result, real_flow)
    rmse, pcc, mae = utils.cal_rmse2(pre_result, real_flow)
    print(cpc)
    print(rmse)
    print(pcc)
    print(mae)


#%%
ori_num = 0
des_num = 500 #1\5--200    3--2000   4\8--400  6--300   7--100  9\0--500

import Visualization_prediction_scatter_bin_function
# Visualization_prediction_scatter_bin_function.draw_scatter_bin(region, "GM-level_single_combine model", 10, real_flow, pre_result, ori_num, des_num)

Visualization_prediction_scatter_bin_function.draw_scatter_bin(region, "GM-level_global model", 10, real_flow, pre_result, ori_num, des_num)

# Visualization_prediction_scatter_bin_function.draw_scatter_bin(region, "GM-level"+str(leve)+" model", 10, real_flow, pre_result, ori_num, des_num)
# Visualization_prediction_scatter_bin_function.draw_scatter_bin(region, "GM model", 10, real_flow, pre_result, ori_num, des_num)

#%%
# import matplotlib.pyplot as plt
# import numpy as np
# font_eng = {'family': 'Arial', 'weight': 'normal', 'size': 18}
# figure, axs = plt.subplots(figsize=(8, 6), dpi=360)
# axs.tick_params(axis='x',length =4,width=1, which='minor',top='on') #,top=True
# axs.tick_params(axis='x',length =8, width=1, which='major',top='on')#,right=True
# axs.tick_params(axis='y',length =8,width=1, which='major', right='on') #,top=True
# axs.tick_params(axis='y',length =4,width=1, which='minor', right='on') #,right=True
# axs.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
# axs.set_xlabel('Seg level',fontdict=font_eng) #设置x轴名称
# axs.set_ylabel('Estimation error',fontdict=font_eng) #设置y轴名称
# axs.tick_params(axis='x', labelsize=14)
# axs.tick_params(axis='y', labelsize=14)
# # axs.set_yscale('log')
# labels=['All','Level 1','Level 2','Level 3','Level 4','Level 5']
# axs.set_ylim(-30, 20)
# # flierprops = dict(marker='.', markerfacecolor='#ebebeb', markersize=5,color='gray')
# box= axs.boxplot(abs_error, labels = labels, patch_artist=True, \
#                  boxprops = {'color':'black','facecolor':'#9999ff'}, \
#                  medianprops = {'linestyle':'--','color':'black'},
#                  flierprops = {'marker':'+','markerfacecolor':'#d3d3d3','markeredgecolor':'#48484a'}) #描点上色
# # plt.setp(box['fliers'], **flierprops)
# plt.show() #展示
#
# figure.savefig('./Figure/box_'+region+'.png')  # 将绘制的图形保存为p4.png