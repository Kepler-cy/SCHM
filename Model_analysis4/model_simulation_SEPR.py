import pandas as pd
import numpy as np
import json
import utils
import geopandas as gpd
state_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_state.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding='gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding='gb18030')
cbsa_name_list_all = ['NY','GLA','WB','SFB', 'GB','DV','AL', 'MM', 'PS',  'MSP']

region = cbsa_name_list_all[9]
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
    location_segregation_value_all = json.load(fp=f)  
for key, v in location_segregation_value_all.items():
    location_segregation_value_all[key] = 1 - v
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

##%%  基于出行数据，筛选研究区域的tess
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

scale_rate = 1/3
tract_pro2_ensemble_gravity = pd.read_csv("./dataset/sepr_data/tract_pro2_ensemble_gravity_"+str(region)+".csv",index_col=0).values
tract_pro2_ensemble_gravity2 = tract_pro2_ensemble_gravity.copy()

for i in range(len(tract_pro2_ensemble_gravity2)):
    if np.max(tract_pro2_ensemble_gravity[i])==np.min(tract_pro2_ensemble_gravity[i]):
        tract_pro2_ensemble_gravity2[i] = (tract_pro2_ensemble_gravity2[i])/(np.max(tract_pro2_ensemble_gravity[i]))
    else:
        tract_pro2_ensemble_gravity2[i] = (tract_pro2_ensemble_gravity2[i]-np.min(tract_pro2_ensemble_gravity[i]))/(np.max(tract_pro2_ensemble_gravity[i])-np.min(tract_pro2_ensemble_gravity[i]))

individual_history_memory = {}
for key, value in user_in_MSA_final.items():
    individual_history_memory[key] = np.zeros((len(subkey_set)))
    for i in range(1, int(len(value) * scale_rate)):
        o_county_id = str(value[i - 1][4]) + str(value[i - 1][5]) + str(value[i - 1][7])
        d_county_id = str(value[i][4]) + str(value[i][5]) + str(value[i][7])
        if (o_county_id not in location_segregation_value.keys()) or (d_county_id not in location_segregation_value.keys()):
            continue
        if o_county_id==d_county_id:
            if i==1:
                individual_history_memory[key][subkey_set_rev[o_county_id]] += 1
            continue
        individual_history_memory[key][subkey_set_rev[d_county_id]] += 1

#%%
oi_threshold = pd.read_csv("./dataset/sepr_data/"+str(region)+"_threshold.csv",index_col=0).values
kq=0
individual_simulation_traj_sepr = {}
for key, value in user_in_MSA_final.items():
    od_clvi_pro = pd.read_csv("./model_results/CLVI_results/CLVI_adj_" + str(region) + "2.csv", index_col=0).values
    tract_pro1_ensemble_gravity = pd.read_csv("./dataset/sepr_data/tract_pro1_ensemble_gravity_" + str(region) + "2.csv", index_col=0).values

    ind_i_history_memory_temp = individual_history_memory[key].copy()
    n_visited_locations = len(np.nonzero(ind_i_history_memory_temp.reshape(-1))[0]) 
    if n_visited_locations==0:
        continue
    for i in range(int(len(value)*scale_rate)):
        current_location_id =  str(value[int(len(value)*scale_rate)-i-1][4]) + \
                               str(value[int(len(value)*scale_rate)-i-1][5]) + str(value[int(len(value)*scale_rate)-i-1][7])
        if current_location_id in location_segregation_value.keys():
            break
    individual_simulation_traj_sepr[key] = [] #set 

    time_s = 0
    current_location_id_ind = current_location_id
    for move_index in range(int(len(value)*scale_rate), len(value)):
        next_location_id_ind = str(value[move_index][4]) + str(value[move_index][5]) + str(value[move_index][7])
        if next_location_id_ind in location_segregation_value.keys() and next_location_id_ind!=current_location_id_ind: #
            time_s+=1
            current_location_id_ind = next_location_id_ind
    current_location_id_ind = current_location_id_ind+"XXXX"

    for move_index in range(time_s):
        od_clvi_pro_nor = od_clvi_pro / od_clvi_pro.sum(axis=1, keepdims=True)
        seg_oo = location_segregation_value[current_location_id]
        for leve in range(5):
            if seg_oo > oi_threshold[leve] or leve == 4:
                oi_seg_dis= od_clvi_pro_nor[leve]
                break
        seg_region_selection = np.random.choice(np.arange(5), size=1, p=oi_seg_dis)[0]
        p_select = oi_seg_dis[seg_region_selection]
        tract_pro1_ensemble_gravity_nor = tract_pro1_ensemble_gravity / tract_pro1_ensemble_gravity.sum(axis=1, keepdims=True)

        o_to_all_travel_pro_matrix1 = tract_pro1_ensemble_gravity_nor[subkey_set_rev[current_location_id]].copy() 
        o_to_all_travel_pro_matrix_select = o_to_all_travel_pro_matrix1 * np.where(np.isclose(np.array(o_to_all_travel_pro_matrix1), p_select), 1, 0)  # 不去那些已经去过的地方
        o_to_all_travel_pro_matrix_select[subkey_set_rev[current_location_id]] = 0  # 

        o_to_all_travel_pro_matrix2 = tract_pro2_ensemble_gravity2[subkey_set_rev[current_location_id]].copy()
        ind_i_history_memory_temp = individual_history_memory[key].copy()
        ind_i_history_memory_temp_reshape = ind_i_history_memory_temp.reshape(-1).copy()
        ind_i_history_memory_temp_reshape[subkey_set_rev[current_location_id]] = 0 #
        ss_his = np.sum(ind_i_history_memory_temp_reshape)
        if  ss_his!= 0:
            his_memory = ind_i_history_memory_temp_reshape / ss_his
        else: # 
            his_memory = np.ones((len(ind_i_history_memory_temp_reshape)))
            his_memory[subkey_set_rev[current_location_id]] = 0
            his_memory /= np.sum(his_memory)

        pp1 = o_to_all_travel_pro_matrix_select * o_to_all_travel_pro_matrix2.reshape(-1)
        pp1 = utils.softmax(pp1.reshape(1, -1)).T
        p_new = np.random.uniform(0, 1)
        if p_new <= 0.95:
            pp = his_memory * pp1.reshape(-1)
        else:
            pp = pp1.reshape(-1)
        pp[subkey_set_rev[current_location_id]] = 0
        if np.sum(pp) != 0:
            pp = pp / np.sum(pp)
        if len(np.nonzero(pp)[0]) == 0:  # 
            kq+=1
            pp = np.ones((len(subkey_set)))  # 
            pp[subkey_set_rev[current_location_id]] = 0
            pp /= np.sum(pp)

        locations = np.arange(len(subkey_set))
        next_location_id = np.random.choice(locations, size=1, p=pp.reshape(-1))[0]
        # 
        ind_move = []
        ind_move.append(current_location_id)
        ind_move.append(subkey_set[next_location_id][0])
        individual_simulation_traj_sepr[key].append(ind_move)
        individual_history_memory[key][next_location_id] += 1
        current_location_id = subkey_set[next_location_id][0]
        seg_dd = location_segregation_value[subkey_set[next_location_id][0]]
        for leve2 in range(5):
            if seg_dd > oi_threshold[leve2] or leve2 == 4:
                od_clvi_pro[leve][leve2] +=1
        one_vec = np.ones(len(tract_pro1_ensemble_gravity)) * np.where(np.isclose(np.array(tract_pro1_ensemble_gravity[subkey_set_rev[current_location_id]]), \
                                                                                  tract_pro1_ensemble_gravity[subkey_set_rev[current_location_id]][next_location_id]), 1, 0)
        tract_pro1_ensemble_gravity[subkey_set_rev[current_location_id]] += one_vec

with open('./model_results/sepr_simulation_results/individual_simulation_traj_sepr_'+str(region)+'2.json', 'w') as f:
    json.dump(individual_simulation_traj_sepr, f)