import pandas as pd
import numpy as np
import json
import geopandas as gpd
import utils
import pickle
state_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_state.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding='gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding='gb18030')
cbsa_name_list_all = ['NY','GLA','WB','SFB', 'GB','DV','AL', 'MM', 'PS',  'MSP']

#%%区域选择
region = cbsa_name_list_all[9]
# [基于bsa筛选区域，county、tract、group]
dataset_name = 'Weeplace_'+region
cbsa_set = pd.read_csv("./dataset/Region/"+str(region)+".csv", header=None)  # cbsa region set
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

tess_tract_selected = tess_tract_selected.reset_index(drop=True); tess_tract_selected = tess_tract_selected.to_crs("epsg:4269");
tess_tract_selected.crs
tess_group_selected = tess_group_selected.reset_index(drop=True); tess_group_selected = tess_group_selected.to_crs("epsg:4269");
tess_group_selected.crs

#%% 数据加载, 转化为od flow形式，并存储flow seg 和 loc name 至 value_flow_seg_name_list
user_in_MSA_final = {}
with open('./dataset/' + str(dataset_name) + '/user_travel_in_' + str(region) + '.json', 'r', encoding='UTF-8') as f:
    user_in_MSA_final = json.load(fp=f)
location_segregation_value = {}
with open('./dataset/' + str(dataset_name) + '/Location_segregation_value_in_region_' + str(region) + '1.json', 'r',
          encoding='UTF-8') as f:
    location_segregation_value = json.load(fp=f)
for key, v in location_segregation_value.items():
    location_segregation_value[key] = 1 - v # 0表示不隔离,1表示隔离
location_segregation_value_all = {}
with open('./dataset/' + str(dataset_name) + '/All_locations_segregation_value1.json', 'r', encoding='UTF-8') as f:
    location_segregation_value_all = json.load(fp=f)  # 部分location是不在目标区域内的
for key, v in location_segregation_value_all.items():
    location_segregation_value_all[key] = 1 - v  # 0表示不隔离,1表示隔离
user_home_info_all_final = {}
with open('./dataset/' + str(dataset_name) + '/user_home_info_all_final.json', 'r', encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f)

odflow_msa = {}
msa_set = {}
for key, value in user_in_MSA_final.items():
    for i in range(1, len(value)):
        o_county_id = str(value[i - 1][4]) + str(value[i - 1][5]) + str(value[i - 1][7])
        d_county_id = str(value[i][4]) + str(value[i][5]) + str(value[i][7])
        if (o_county_id not in location_segregation_value.keys()) or (d_county_id not in location_segregation_value.keys()):
            continue
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

#%%  基于出行数据，筛选研究区域的tess
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
tess_selected_final = tess_selected_final.reset_index(drop=True);
tess_selected_final = tess_selected_final.to_crs("epsg:4269");
tess_selected_final.crs

subkey_set,subkey_set_rev = {},{}
k=0
for i in range(len(tess_tract_selected)):
    if str(tess_tract_selected.iloc[i].GEOID) not in location_segregation_value.keys():
        continue
    subkey_set[k] = []
    subkey_set[k].append(str(tess_tract_selected.iloc[i].GEOID))
    subkey_set[k].append(tess_tract_selected.iloc[i].POPULATION)
    subkey_set[k].append(tess_tract_selected.iloc[i].geometry.centroid.x)
    subkey_set[k].append(tess_tract_selected.iloc[i].geometry.centroid.y)
    subkey_set_rev[str(tess_tract_selected.iloc[i].GEOID)] = k
    k+=1

od_clvi_pro = pd.read_csv("./model_results/CLVI_results/CLVI_adj_"+str(region)+"2.csv",index_col=0).values
tract_pro1_ensemble_gravity = np.zeros((len(subkey_set),len(subkey_set)))
oi_threshold = pd.read_csv("./dataset/sepr_data/"+str(region)+"_threshold.csv",index_col=0).values

for i in range(len(subkey_set)): #
    if i%10==0:
        print(i)
    seg_i = location_segregation_value[subkey_set[i][0]]
    for leve in range(5):
        if seg_i > oi_threshold[leve] or leve==4:
            oi_seg_level_num = leve
            break
    for j in range(len(subkey_set)):
        if i==j:
            tract_pro1_ensemble_gravity[i][j] = od_clvi_pro[oi_seg_level_num][oi_seg_level_num]
            continue
        seg_j = location_segregation_value[subkey_set[j][0]]
        for leve in range(5):
            if seg_j > oi_threshold[leve] or leve == 4:
                di_seg_level_num = leve
                break
        tract_pro1_ensemble_gravity[i][j] = od_clvi_pro[oi_seg_level_num][di_seg_level_num]

tract_pro1_ensemble_gravity_pd = pd.DataFrame(tract_pro1_ensemble_gravity)
tract_pro1_ensemble_gravity_pd.to_csv("./dataset/sepr_data/tract_pro1_ensemble_gravity_"+str(region)+"2.csv")