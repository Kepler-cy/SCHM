import pandas as pd
import numpy as np
import json
import geopandas as gpd
import matplotlib as mpl
from matplotlib.collections import LineCollection
import copy
import draw_utils
import math
import utils
from copy import copy
import matplotlib.pyplot as plt
state_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_state.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding='gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding='gb18030')
cbsa_name_list_all = ['NY', 'GLA', 'WB', 'SFB', 'GB', 'DV', 'AL', 'MM', 'PS', 'MSP']

#%% [基于bsa筛选区域，county、tract、group]
model_name = "radiation"
region = cbsa_name_list_all[9]
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

#  数据加载, 转化为od flow形式，并存储flow seg 和 loc name 至 value_flow_seg_name_list
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

# 统计 od flow data
odflow_msa = {}
msa_set = {}
for key, value in user_in_MSA_final.items():
    # o_county_id = user_home_info_all_final[key][:-1]
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

#%% 生成 Null model
o2d2flow_msa_null_model = {}
for key, values in o2d2flow_msa.items():
    o2d2flow_msa_null_model[key] = {}
    #重复仿真N次
    for i in range(o_flow_sum[key]):
        location_id = np.random.choice(len(subkey_set), size=1, p=(np.ones(len(subkey_set))/len(subkey_set)).reshape(-1))[0]
        if subkey_set[location_id][0] not in o2d2flow_msa_null_model[key].keys():
            o2d2flow_msa_null_model[key][subkey_set[location_id][0]] = 1
        else:
            o2d2flow_msa_null_model[key][subkey_set[location_id][0]] += 1

odflow_msa_null_model = {}
for key, values in o2d2flow_msa_null_model.items():
    for subkey, va in values.items():
        if (str(key), str(subkey)) not in odflow_msa_null_model.keys():
            odflow_msa_null_model[(str(key), str(subkey))] = 1
        else:
            odflow_msa_null_model[(str(key), str(subkey))] += 1

#%% 生成 gravity model
tract_pro = np.zeros((len(tess_tract_selected),len(tess_tract_selected)))
for i in range(len(tess_tract_selected)):
    if i%500==0:
        print(i)
    # origin
    o_id = subkey_set[i][0]
    pop_o = subkey_set[i][1]
    for j in range(len(tess_tract_selected)):
        if i == j or pop_o<=0:
            tract_pro[i][j]=0
            continue
        d_id = subkey_set[j][0]
        pop_d = subkey_set[i][1]
        dis_od = utils.earth_distance([subkey_set[i][2],subkey_set[i][3]], [subkey_set[j][2],subkey_set[j][3]])
        pro = (pop_o *pop_d) * (dis_od ** (-2))
        tract_pro[i][j] = pro
    sum_ = np.sum(tract_pro[i])
    if sum_==0:
        continue
    for j in range(len(tess_tract_selected)):
        tract_pro[i][j] /= sum_

o2d2flow_msa_gravity_model = {}
for key, values in o2d2flow_msa.items():
    o2d2flow_msa_gravity_model[key] = {}
    #重复仿真N次
    for i in range(o_flow_sum[key]):
        pp = tract_pro[subkey_set_rev[key]]
        if sum(pp)!=1 and sum(pp)==0:
            tract_pro[subkey_set_rev[key]] = np.ones(len(tract_pro))/len(tract_pro)
        location_id = np.random.choice(len(subkey_set), size=1, p=pp.reshape(-1))[0]
        if subkey_set[location_id][0] not in o2d2flow_msa_gravity_model[key].keys():
            o2d2flow_msa_gravity_model[key][subkey_set[location_id][0]] = 1
        else:
            o2d2flow_msa_gravity_model[key][subkey_set[location_id][0]] += 1

odflow_msa_gravity_model = {}
for key, values in o2d2flow_msa_gravity_model.items():
    for subkey, va in values.items():
        if (str(key), str(subkey)) not in o2d2flow_msa_gravity_model.keys():
            odflow_msa_gravity_model[(str(key), str(subkey))] = 1
        else:
            odflow_msa_gravity_model[(str(key), str(subkey))] += 1

#%% 生成 radiation model
import time, operator
tract_pro = np.zeros((len(tess_tract_selected),len(tess_tract_selected)))
for o_id in range(len(tess_tract_selected)): #针对每一个i，都有遍历所有的目的地，得到697*697的二维矩阵
    origin_relevance = subkey_set[o_id][1] #POP_i
    if origin_relevance==0:
        continue
    normalization_factor = 1.0
    destinations_and_distances = []
    for d_id in range(len(tess_tract_selected)): #遍历所有的D
        if d_id != o_id:
            destinations_and_distances +=  [(d_id, utils.earth_distance([subkey_set[o_id][2],subkey_set[o_id][3]], [subkey_set[d_id][2],subkey_set[d_id][3]]))]
    # sort the destinations by distance (from the closest to the farthest)
    destinations_and_distances.sort(key=operator.itemgetter(1))
    sum_inside = 0.0
    for d_id, dis_ in destinations_and_distances:
        destination_relevance = subkey_set[d_id][1]
        if (origin_relevance + sum_inside)==0:
            prob_origin_destination = 0.0
        else:
            prob_origin_destination = normalization_factor * \
                                  (origin_relevance * destination_relevance) / \
                                  ((origin_relevance + sum_inside) * (origin_relevance + sum_inside + destination_relevance))
        sum_inside += destination_relevance
        tract_pro[o_id][d_id] = prob_origin_destination

o2d2flow_msa_radiation_model = {}
for key, values in o2d2flow_msa.items():
    o2d2flow_msa_radiation_model[key] = {}
    #重复仿真N次
    pp = tract_pro[subkey_set_rev[key]]
    if sum(pp) == 0:
        tract_pro[subkey_set_rev[key]] = np.ones(len(tract_pro)) / len(tract_pro)
    elif sum(pp) != 1:
        pp = pp / sum(pp)
    for i in range(o_flow_sum[key]):
        location_id = np.random.choice(len(subkey_set), size=1, p=pp.reshape(-1))[0]
        if subkey_set[location_id][0] not in o2d2flow_msa_radiation_model[key].keys():
            o2d2flow_msa_radiation_model[key][subkey_set[location_id][0]] = 1
        else:
            o2d2flow_msa_radiation_model[key][subkey_set[location_id][0]] += 1

odflow_msa_radiation_model = {}
for key, values in o2d2flow_msa_radiation_model.items():
    for subkey, va in values.items():
        if (str(key), str(subkey)) not in o2d2flow_msa_radiation_model.keys():
            odflow_msa_radiation_model[(str(key), str(subkey))] = 1
        else:
            odflow_msa_radiation_model[(str(key), str(subkey))] += 1

#%% select model
if model_name =="null":
    o2d2flow_msa_model = o2d2flow_msa_null_model.copy()
    odflow_msa_model = odflow_msa_null_model.copy()
elif model_name =="gravity":
    o2d2flow_msa_model = o2d2flow_msa_gravity_model.copy()
    odflow_msa_model = odflow_msa_gravity_model.copy()
elif model_name =="radiation":
    o2d2flow_msa_model = o2d2flow_msa_radiation_model.copy()
    odflow_msa_model = odflow_msa_radiation_model.copy()

# 存储flow\seg\loc name value
value_flow_seg_name_list = []
for key, values in o2d2flow_msa_model.items():
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
for key, values in o2d2flow_msa_model.items():
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
    print(dy);
    print(dx);
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
    # plt.savefig("./Figure/Seg_value_of_different_seg_levels_all_" + str(region)+'_'+str(model_name) +".png", dpi=360,bbox_inches='tight')
    plt.show()
dd_all()

# 网络特征与个体出行特征
max_level_num = 5
loc_level_degree_all, loc_level_distance_all, loc_level_poiclass_all = [],[],[]
for i in range(max_level_num):
    loc_level_degree_all.append([])
    loc_level_distance_all.append([])
    loc_level_poiclass_all.append({})

#计算不同level的loc出发的dis、degree
for key, va in o2d2flow_msa_model.items():
    if key not in location_segregation_value_all.keys():
        continue
    o_seg = location_segregation_value_all[key]
    o_lng_lat = [subkey_set[subkey_set_rev[key]][2],subkey_set[subkey_set_rev[key]][3]]
    for i in range(max_level_num):
        if i==max_level_num-1:
            for subkey, va in o2d2flow_msa_model[key].items():
                d_lng_lat = [subkey_set[subkey_set_rev[subkey]][2],subkey_set[subkey_set_rev[subkey]][3]]
                dis = utils.earth_distance(o_lng_lat, d_lng_lat)
                loc_level_distance_all[max_level_num-1].append(dis)
            loc_level_degree_all[max_level_num-1].append(len(o2d2flow_msa_model[key]))
        else:
            if o_seg > x_intersection[i]:
                for subkey, va in o2d2flow_msa_model[key].items():
                    d_lng_lat = [subkey_set[subkey_set_rev[subkey]][2],subkey_set[subkey_set_rev[subkey]][3]]
                    dis = utils.earth_distance(o_lng_lat, d_lng_lat)
                    loc_level_distance_all[i].append(dis)
                loc_level_degree_all[i].append(len(o2d2flow_msa_model[key]))
                break

font1 = {'family' : 'Arial','color':'Black','size': 16}
def draw_distribution_pre(datax,datay,dis_name,region):
    #设置图片的大小
    figsize = 6,6
    plt.rcParams['xtick.direction'] = 'in' ####坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in' ####坐标轴刻度朝内
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='x',length =4,width=1, which='minor',top='on') #,top=True
    ax.tick_params(axis='x',length =8, width=1, which='major',top='on')#,right=True
    ax.tick_params(axis='y',length =8,width=1, which='major', right='on') #,top=True
    ax.tick_params(axis='y',length =4,width=1, which='minor', right='on') #,right=True
    ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    sh=plt.gca()
    ax.set_xscale("log")
    # ax.set_yscale("log")
    if dis_name=="Traveling distance":
        sh.set_xlabel("$\it{d}$ (km)",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{d}$)",fontdict=font1) #纵轴名称
        mm_arker = 'o'
    elif dis_name=="Visiting number":
        # sh.set_xlabel("$\it{v}$",fontdict=font1) #横轴名称
        sh.set_xlabel("Visiting number",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{v}$)",fontdict=font1) #纵轴名称
        mm_arker = '*'
    elif dis_name=="Traveling step":
        sh.set_xlabel("$\it{L}$",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{L}$)",fontdict=font1) #纵轴名称
        mm_arker = 'D'
    elif dis_name=="Visiting frequency":
        sh.set_xlabel("Visiting frequency",fontdict=font1) #横轴名称
        # sh.set_xlabel("$\it{f}$",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{f}$)",fontdict=font1) #纵轴名称
        mm_arker = 'd'
        # ax.set_xlim(10, 1000)
    elif dis_name=="Traveling radius":
        sh.set_xlabel("Traveling radius",fontdict=font1) #横轴名称
        # sh.set_xlabel("$\it{r}$ (km)",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{r}$)",fontdict=font1) #纵轴名称
        mm_arker = '^'
    elif dis_name=="Traveling degree":
        sh.set_xlabel("Degree",fontdict=font1) #横轴名称
        # sh.set_xlabel("$\it{r}$ (km)",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{de}$)",fontdict=font1) #纵轴名称
        mm_arker = '^'
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    color_s = ['orange','#888888','tab:purple','#aadb3d','royalblue']
    marker_s = ['s','^','h','*','o']
    for i in range(5):
        plt.plot(datax[i],datay[i],linewidth=1.2,c=color_s[i],label="Seg level "+str(i),marker=marker_s[i],markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    plt.subplots_adjust(left=0,right=0.9)
    plt.legend()
    plt.savefig("./Figure/Data_distribution "+str(dis_name)+"_"+str(region)+".png",dpi=360,bbox_inches = 'tight')

def cal_his_figure(dataset, numBins):
    fig = plt.figure(figsize=(6, 4))
    # x_pad_width= width #0.2 # 分组柱状图中两个组之间的距离
    ax = fig.add_subplot(1, 1, 1)
    # data = subplot_values
    ax.tick_params(axis='x', length=4, width=1, which='major', bottom='on')  # ,top=True
    ax.tick_params(axis='x', length=4, width=1, which='major', top='on')  # ,right=True
    ax.tick_params(axis='y', length=8, width=1, which='major', left='on')  # ,top=True
    ax.tick_params(axis='y', length=4, width=1, which='major', right='on')  # ,right=True
    ax.spines['bottom'].set_linewidth(1);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);  ####设置上部坐标轴的粗细

    # 设置刻度线标识的字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontsize(13) for label in labels]

    ax = fig.add_subplot(111)
    fre_real, bins_real, patches = ax.hist(dataset, numBins, color='blue', alpha=0.4, rwidth=0.7, edgecolor='black',
                                           align='left', histtype='bar')
    bin_real = np.zeros(numBins)
    for i in range(len(bins_real) - 1):
        bin_real[i] = (bins_real[i] + bins_real[i + 1]) / 2.0
    sum_ = np.sum(fre_real)
    for i in range(len(fre_real)):
        fre_real[i] = fre_real[i] / sum_
    return bin_real, fre_real

bbin = 15
bin_loc_level_degree_all,fre_loc_level_degree_all = [],[]
for i in range(len(loc_level_degree_all)):
    bin_loc_level_degree_ind,fre_loc_level_degree_ind  = cal_his_figure(loc_level_degree_all[i], bbin)
    bin_loc_level_degree_all.append(bin_loc_level_degree_ind)
    fre_loc_level_degree_all.append(fre_loc_level_degree_ind)
draw_distribution_pre(bin_loc_level_degree_all,fre_loc_level_degree_all,"Traveling degree",region+"_"+str(model_name)+"_model")

bbin = 10
bin_loc_level_distance_all,fre_loc_level_distance_all = [],[]
for i in range(len(loc_level_distance_all)):
    bin_loc_level_distance_ind,fre_loc_level_distance_ind  = cal_his_figure(loc_level_distance_all[i], bbin)
    bin_loc_level_distance_all.append(bin_loc_level_distance_ind)
    fre_loc_level_distance_all.append(fre_loc_level_distance_ind)
draw_distribution_pre(bin_loc_level_distance_all,fre_loc_level_distance_all,"Traveling distance",region+"_"+str(model_name)+"_model")

bin_loc_level_degree_all_pd = pd.DataFrame(bin_loc_level_degree_all)
fre_loc_level_degree_all_pd = pd.DataFrame(fre_loc_level_degree_all)
bin_loc_level_degree_all_pd.to_csv("./model_results/Travel_features_value/bin_level_degree_all_"+str(model_name)+"_"+str(region)+".csv")
fre_loc_level_degree_all_pd.to_csv("./model_results/Travel_features_value/fre_level_degree_all_"+str(model_name)+"_"+str(region)+".csv")
bin_loc_level_distance_all_pd = pd.DataFrame(bin_loc_level_distance_all)
fre_loc_level_distance_all_pd = pd.DataFrame(fre_loc_level_distance_all)
bin_loc_level_distance_all_pd.to_csv("./model_results/Travel_features_value/bin_level_distance_all_"+str(model_name)+"_"+str(region)+".csv")
fre_loc_level_distance_all_pd.to_csv("./model_results/Travel_features_value/fre_level_distance_all_"+str(model_name)+"_"+str(region)+".csv")

#%%
# 分析5个level之间的访问模式，热图矩阵呈现
seg_to_seg_adj = np.zeros((5, 5))
seg_to_seg_dis_adj = np.zeros((5, 5))
kk=0
for (o_id, d_id), va in odflow_msa_model.items(): #计算2个seg_level之间的dis、flow，然后写入adj矩阵中
    if (o_id not in location_segregation_value_all.keys()) or (d_id not in location_segregation_value_all.keys()):
        kk+=1
        continue
    o_seg, d_seg = location_segregation_value_all[o_id], location_segregation_value_all[d_id]
    o_lng_lat = [subkey_set[subkey_set_rev[o_id]][2],subkey_set[subkey_set_rev[o_id]][3]]
    d_lng_lat = [subkey_set[subkey_set_rev[d_id]][2],subkey_set[subkey_set_rev[d_id]][3]]
    # o_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == o_id].geometry.centroid.values[0]
    # d_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == d_id].geometry.centroid.values[0]
    dis = utils.earth_distance(o_lng_lat, d_lng_lat)
    level_o, level_d = 0, 0
    for i in range(max_level_num):
        if o_seg > x_intersection[i]:
            level_o = i
            break
        if i == max_level_num - 1:
            level_o = i
    for i in range(max_level_num):
        if d_seg > x_intersection[i]:
            level_d = i
            break
        if i == max_level_num - 1:
            level_d = i
    seg_to_seg_adj[level_o][level_d] += va
    seg_to_seg_dis_adj[level_o][level_d] += va * dis

# 归一化-可视化
seg_to_seg_adj_save = seg_to_seg_adj.copy()
seg_to_seg_dis_adj_save = seg_to_seg_dis_adj.copy()

seg_to_seg_adj_pd = pd.DataFrame(seg_to_seg_adj)
seg_to_seg_dis_adj_pd = pd.DataFrame(seg_to_seg_dis_adj)
seg_to_seg_adj_pd.to_csv("./model_results/CLVI_results/CLVI_adj_"+str(model_name)+"_"+str(region)+".csv")
seg_to_seg_dis_adj_pd.to_csv("./model_results/CLVI_results/CLVI_dis_adj_"+str(model_name)+"_"+str(region)+".csv")

for i in range(5):
    for j in range(5):
        if seg_to_seg_adj_save[i][j] != 0:
            seg_to_seg_dis_adj_save[i][j] = seg_to_seg_dis_adj_save[i][j] / seg_to_seg_adj_save[i][j]

seg_to_seg_adj_save_nor = seg_to_seg_adj_save.copy()
for i in range(5):
    # for j in range(5):
    #     if seg_to_seg_adj_save_nor[i][j]!=0:
    #         seg_to_seg_adj_save_nor[i][j] = np.log(seg_to_seg_adj_save_nor[i][j])
    seg_to_seg_adj_save_nor[:][i] = seg_to_seg_adj_save_nor[:][i] / sum(seg_to_seg_adj_save_nor[:][i])

seg_to_seg_dis_adj_save_nor = seg_to_seg_dis_adj_save.copy()
for i in range(5):
    seg_to_seg_dis_adj_save_nor[i] = seg_to_seg_dis_adj_save_nor[i] / sum(seg_to_seg_dis_adj_save_nor[i])

# 可视化各个block 4个level的比例
def draw_heatmap(data, name, cc):
    font1 = {'family': 'Arial', 'color': 'Black', 'size': 18}
    f, ax = plt.subplots(figsize=(9, 8))
    plt.imshow(data, cmap=cc, aspect=1)
    cb = plt.colorbar(shrink=0.9)
    cb.ax.tick_params(labelsize=17)  # 设置色标刻度字体大小。
    sh = plt.gca()
    x = range(0, 5)
    plt.xticks(x)
    plt.xticks(fontsize=17)  # 横轴刻度字体大小
    plt.yticks(fontsize=17)  # 纵轴刻度字体大小
    sh.set_xlabel("D_Levels", fontdict=font1)  # 横轴名称
    sh.set_ylabel("O_Levels", fontdict=font1)  # 纵轴名称
    plt.savefig("./Figure/Seg_to_seg_adj_" + str(name) +"_"+str(model_name)+"_model.png", dpi=360, bbox_inches='tight')
    # plt.show()

draw_heatmap(seg_to_seg_adj_save_nor, "seg_adj_nor", 'Reds')
draw_heatmap(seg_to_seg_dis_adj_save_nor, "dis_adj_nor", 'Reds')

seg_to_seg_adj_save_nor_pd = pd.DataFrame(seg_to_seg_adj_save_nor)
seg_to_seg_adj_save_nor_pd.to_csv("./model_results/CLVI_results/CLVI_adj_nor_"+str(model_name)+"_"+str(region)+".csv")
seg_to_seg_dis_adj_save_nor_pd = pd.DataFrame(seg_to_seg_dis_adj_save_nor)
seg_to_seg_dis_adj_save_nor_pd.to_csv("./model_results/CLVI_results/CLVI_dis_adj_nor_"+str(model_name)+"_"+str(region)+".csv")

bsa_seg_index = 0 #overall访问倾向
for i in range(5):
    for j in range(5):
        bsa_seg_index += seg_to_seg_adj_save_nor[i][j]*(i-j)

bsa_seg_index_upper = 0 #往隔离程度低的地方的访问倾向
for i in range(5):
    for j in range(5):
        if i < j:
            bsa_seg_index_upper += seg_to_seg_adj_save_nor[i][j]*(i-j)

bsa_seg_index_lower = 0 #往隔离程度高的地方的访问倾向
for i in range(5):
    for j in range(5):
        if i > j:
            bsa_seg_index_lower += seg_to_seg_adj_save_nor[i][j]*(i-j)

print(bsa_seg_index)
print(bsa_seg_index_upper)
print(bsa_seg_index_lower)

#%%  分析5个level之间的Shannon entropy
out_flow_level_all, out_flow_all = [],[]
se_list_level_all, se_list_all = [], []
for i in range(max_level_num):
    out_flow_level_all.append([])
    se_list_level_all.append([])

for key, values in o2d2flow_msa_model.items():
    if key in location_segregation_value.keys():
        seg_v = location_segregation_value[key]
        sum_ = 0
        for subkey, va in o2d2flow_msa_model[key].items():
            sum_ += va
        ind = []
        for subkey, va in o2d2flow_msa_model[key].items():
            ind.append(va / sum_)
        out_flow_all.append(ind)
        for i in range(max_level_num):
            if seg_v > x_intersection[i]:
                out_flow_level_all[i].append(ind)
                break
            if i== max_level_num-1:
                out_flow_level_all[i].append(ind)

for level_num in range(max_level_num):
    for i in range(len(out_flow_level_all[level_num])):
        se = 0
        for j in range(len(out_flow_level_all[level_num][i])):
            se += -1.0 * out_flow_level_all[level_num][i][j] * np.log(out_flow_level_all[level_num][i][j])
        se_list_level_all[level_num].append(se)

for i in range(len(out_flow_all)):
    se = 0
    for j in range(len(out_flow_all[i])):
        se += -1.0 * out_flow_all[i][j] * np.log(out_flow_all[i][j])
    se_list_all.append(se)

def get_cdf(data):
    data_fre = pd.Series(data).value_counts()
    data_fre_sort = data_fre.sort_index(axis=0, ascending=True)
    data_fre_sort_df = data_fre_sort.reset_index()
    data_fre_sort_df[0] = data_fre_sort_df[0] / len(data)  # 将频数转换成概率
    data_fre_sort_df.columns = ['Rds', 'Fre']  # 将列表列索引重命名
    data_fre_sort_df['cumsum'] = np.cumsum(data_fre_sort_df['Fre'])
    return data_fre_sort_df

se_list_fre_sort_df_all =[]
for i in range(max_level_num):
    se_list_fre_sort_df_all.append(get_cdf(se_list_level_all[i]))

se_list_all_fre_sort_df = get_cdf(se_list_all)

for i in range(max_level_num):
    ind = se_list_fre_sort_df_all[i]
    ind.to_csv('./model_results/Travel_features_value/Travel_entropy_Level'+str(i)+"_"+str(model_name)+"_"+str(region)+".csv")
se_list_all_fre_sort_df.to_csv('./model_results/Travel_features_value/Travel_entropy_All'+"_"+str(model_name)+"_"+str(region)+".csv")

plot = plt.figure()
ax1 = plot.add_subplot(1, 1, 1)
for i in range(max_level_num):
    ax1.plot(se_list_fre_sort_df_all[i]['Rds'], se_list_fre_sort_df_all[i]['cumsum'], label="Seg level "+str(i))
ax1.plot(se_list_all_fre_sort_df['Rds'], se_list_all_fre_sort_df['cumsum'], color='black', label="All")
plt.xlabel("Entropy $se$")
plt.ylabel("P(seg$_{i}$>$se$)")
plt.legend()
# plt.savefig("./Figure/Entropy_of_different_Seg_levels_" + str(region) + "_"+str(model_name)+"_model.png", dpi=360, bbox_inches='tight')
plt.show()

#  对出行数据进行聚类系数分析
out_flow_level_all, out_flow_all = [], []
c_list_level_all, c_all = [], []
for i in range(max_level_num):
    out_flow_level_all.append([])
    c_list_level_all.append([])

max_,min_ = -1,99999999
for key, values in o2d2flow_msa_model.items():
    if key in location_segregation_value.keys():
        for subkey, va in o2d2flow_msa_model[key].items():
            max_ = max(max_, va)
            min_ = min(min_, va)

od_flow_new = {}
for key, values in o2d2flow_msa_model.items():
    if key in location_segregation_value.keys():
        od_flow_new[key] = {}
        for subkey, va in o2d2flow_msa_model[key].items():
            od_flow_new[key][subkey] = va#/max_

min_, max_ = 999, -1
for key, values in od_flow_new.items():
    seg_v = location_segregation_value[key]
    degree_i, sum_ = len(od_flow_new[key]), 0
    for subkey_j, values_j in od_flow_new[key].items():  # 遍历 j
        if subkey_j in od_flow_new.keys():
            for subkey_k, values_k in od_flow_new[subkey_j].items():  # 遍历 k
                if (subkey_k in od_flow_new.keys()) and (key in od_flow_new[subkey_k].keys()):
                    sum_ += math.pow(od_flow_new[key][subkey_j] * od_flow_new[subkey_j][subkey_k] * od_flow_new[subkey_k][key], 1 / 3)
    if degree_i > 1:
        ci = sum_ / (degree_i * (degree_i - 1))
        c_all.append(ci)
        for i in range(max_level_num):
            if seg_v > x_intersection[i]:
                c_list_level_all[i].append(ci)
                break
            if i==max_level_num-1:
                c_list_level_all[i].append(ci)
        min_ = min(min_, ci)
        max_ = max(max_, ci)

c_list1_fre_sort_df_all =[]
for i in range(max_level_num):
    c_list1_fre_sort_df_all.append(get_cdf(c_list_level_all[i]))

c_all_fre_sort_df = get_cdf(c_all)

for i in range(max_level_num):
    ind = c_list1_fre_sort_df_all[i]
    ind.to_csv('./model_results/Travel_features_value/Clustering_coefficient_Level'+str(i)+"_"+str(model_name)+"_"+str(region)+".csv")
c_all_fre_sort_df.to_csv('./model_results/Travel_features_value/Clustering_coefficient_All'+"_"+str(model_name)+"_"+str(region)+".csv")

# new_row1 = {'Rds': 0,'Fre':0, 'cumsum': 0}
# new_row2 = {'Rds': 0,'Fre':0, 'cumsum': 0}
# 在开头插入新行数据
# c_list1_fre_sort_df = pd.concat([pd.DataFrame([new_row]), c_list1_fre_sort_df], ignore_index=True)
plot = plt.figure()
ax1 = plot.add_subplot(1, 1, 1)
for i in range(max_level_num):
    ax1.plot(c_list1_fre_sort_df_all[i]['Rds'], c_list1_fre_sort_df_all[i]['cumsum'], label="Seg level "+str(i))
ax1.plot(c_all_fre_sort_df['Rds'], c_all_fre_sort_df['cumsum'], color='black', label="All")
plt.xlabel("Clustering coefficient $c$")
plt.ylabel("Cumsum(Coef$_{i}$>$c$)")
# ax1.set_xlim(0, 0.001)
plt.legend()
# plt.savefig("./Figure/Clustering_coefficient_of_different_Seg_levels_" + str(region) + "_"+str(model_name)+"_model.png", dpi=360,
#             bbox_inches='tight')
plt.show()
