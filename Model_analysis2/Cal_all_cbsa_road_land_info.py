import pandas as pd
import numpy as np
import json
import geopandas as gpd
from shapely import geometry
from dateutil import parser
import copy
import datetime
import os
dataset_name = 'Weeplace'

#%% 加载数据
dataset_all = pd.read_csv("./dataset/weeplace_checkins.csv",header=0) #7658368
dataset_all.columns=['uid','placeid','time','lat','lng','city','poi']

#%%
county_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_county.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding = 'gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding = 'gb18030')

#%% 拼接cbas
cbsa_name_list = ['NY','GLA','WB','SFB','DFW', 'GB','GH','DV','AL', 'MM', 'MD', 'PS', 'GO', 'MSP', \
                  'DA', 'NO', 'PLM', 'GSL', 'CC', 'SLC', 'SM', 'PNC', 'SA', 'IM', 'CM', 'RT', 'NM', 'GM','HR']
cbsa_combine = None
for i in range(len(cbsa_name_list)):
    cbsa_name_data = pd.read_csv("./dataset/Region/"+str(cbsa_name_list[i])+".csv", header=None)  # cbsa region set
    cbsa_name_data.insert(cbsa_name_data.shape[1], 'region', str(cbsa_name_list[i]))
    if i==0:
        cbsa_combine = cbsa_name_data
    else:
        cbsa_combine = pd.concat([cbsa_combine, cbsa_name_data])
cbsa_combine.rename(columns={0: "county", 1: "state", 2: "state_id", "region": "CBSA", 3: "none"},inplace=True)
cbsa_combine = cbsa_combine.drop(['none'],axis=1)

#%% 筛选cbsa区域的county\tract和group
for i in range(len(cbsa_combine)): #county tess
    if i==0:
        if cbsa_combine.iloc[i][2]<10:
            aa="0"+str(cbsa_combine.iloc[i][2])
        else:
            aa = str(cbsa_combine.iloc[i][2])
        tess_county_all_selected = county_all[(county_all['STATEFP']==aa) & (county_all['NAMELSAD']==cbsa_combine.iloc[i][0])]
    else:
        if cbsa_combine.iloc[i][2]<10:
            aa="0"+str(cbsa_combine.iloc[i][2])
        else:
            aa = str(cbsa_combine.iloc[i][2])
        tess_county_all_selected = pd.concat((tess_county_all_selected,county_all[(county_all['STATEFP']==aa) & (county_all['NAMELSAD']==cbsa_combine.iloc[i][0])]))
#tract tess
for i in range(len(cbsa_combine)):
    if i==0:
        tess_tract_selected = tess_tract_all[(tess_tract_all['STATENAME']==cbsa_combine.iloc[i][1]) & (tess_tract_all['COUNTYNAME']==cbsa_combine.iloc[i][0])]
    else:
        tess_tract_selected = pd.concat((tess_tract_selected, tess_tract_all[(tess_tract_all['STATENAME']==cbsa_combine.iloc[i][1]) & (tess_tract_all['COUNTYNAME']==cbsa_combine.iloc[i][0])]))
#group tess
for i in range(len(cbsa_combine)):
    if i==0:
        tess_group_selected = tess_all[(tess_all['STATENAME']==cbsa_combine.iloc[i][1]) & (tess_all['COUNTYNAME']==cbsa_combine.iloc[i][0])]
    else:
        tess_group_selected = pd.concat((tess_group_selected,tess_all[(tess_all['STATENAME']==cbsa_combine.iloc[i][1]) & (tess_all['COUNTYNAME']==cbsa_combine.iloc[i][0])]))

start_time = datetime.datetime.now()
tess_tract_all = pd.read_csv('./dataset/mapdata/tess_tract_all.csv',index_col=0)
tess_tract_all = gpd.GeoDataFrame(tess_tract_all)
from shapely import wkt
tess_tract_all['geometry']= tess_tract_all['geometry'].apply(wkt.loads)
end_time = datetime.datetime.now()

#%% 建立字典，统计不同tract的road_density 类别和数目
path1 = './dataset/road_density'
all_state_road_doc_name = os.listdir(path1)
road_density_set = {}
for dir_name in all_state_road_doc_name:
    ind_road_info = gpd.GeoDataFrame.from_file('./dataset/road_density/'+str(dir_name)+'/'+ str(dir_name),encoding='gb18030')
    for i in range(len(ind_road_info)):
        geo_id_ind = str(ind_road_info.iloc[i][3])
        if geo_id_ind not in road_density_set.keys():
            road_density_set[geo_id_ind] = []
            road_density_set[geo_id_ind].append((ind_road_info.loc[i]['LENGTH'])) #length
            road_density_set[geo_id_ind].append((ind_road_info.loc[i]['AREA'])) #area
            road_density_set[geo_id_ind].append((ind_road_info.loc[i]['DENSITY'])) #density

with open('./dataset/road_density_set.json', 'w') as f:
    json.dump(road_density_set, f) #72740

#%% 分层的分配相应的 landuse 到相应的county、tract中
path2 = './dataset/Land_use'
all_state_land_doc_name = os.listdir(path2)
# land_use_set = {}
for dir_name in all_state_land_doc_name[4:5]:
    # dir_name = all_state_land_doc_name[4]
    land_use_set = {}
    ind_land_use_info = gpd.GeoDataFrame.from_file('./dataset/Land_use/'+str(dir_name)+'/'+ 'gis_osm_landuse_a_free_1.shp',encoding='gb18030')
    for i in range(len(ind_land_use_info)):
        land_use_ind_geometry = ind_land_use_info.loc[i]['geometry'].centroid
        if dir_name=='california-latest-free.shp':
            land_use_ind_info = ind_land_use_info.loc[i]['landuse']
        else:
            land_use_ind_info = ind_land_use_info.loc[i]['fclass']
        bound_county = county_all["geometry"]
        county_num, county_name, ind = -1, None, []
        for index, bound in enumerate(bound_county):
            if bound.contains(geometry.Point(land_use_ind_geometry.x, land_use_ind_geometry.y)):
                ind.append(land_use_ind_geometry.y) #lat
                ind.append(land_use_ind_geometry.x) #lng
                ind.append(str(county_all.iloc[index].STATEFP)) #state id
                ind.append(str(county_all.iloc[index].COUNTYFP)) #County id
                ind.append(str(county_all.iloc[index].NAMELSAD)) #County name
                county_num = str(county_all.iloc[index].COUNTYFP)
                county_name = str(county_all.iloc[index].NAMELSAD)
                break
        if county_num == -1:
            continue
        #基于county id，定位相应CBG的地理边界信息，数据对应的是 tess_county_selected
        tract_num = ""
        bound_tract = copy.deepcopy(tess_tract_all[tess_tract_all['COUNTYFP']==county_num])
        bound_tract_geom = bound_tract['geometry']
        for index, bound in enumerate(bound_tract_geom):
            if bound.contains(geometry.Point(land_use_ind_geometry.x, land_use_ind_geometry.y)):
                tract_num = str(bound_tract.iloc[index].TRACTCE)
                ind.append(tract_num)
                break
        if tract_num == "":
            continue
        tract_info = str(ind[2])+str(ind[3])+str(ind[5])#+str(ind[6])
        if tract_info not in land_use_set.keys():
            land_use_set[tract_info] = []
            land_use_set[tract_info].append(land_use_ind_info)
        else:
            land_use_set[tract_info].append(land_use_ind_info)

    with open('./dataset/land_use_select/land_use_set_'+ str(dir_name[:-4]) +'.json', 'w') as f:
        json.dump(land_use_set, f) #352

#%% 处理组合土地信息
path2 = './dataset/land_use_select'
all_land_use_select_doc_name = os.listdir(path2)

land_use_info_combine = {}
for land_use_name in all_land_use_select_doc_name:
    with open('./dataset/land_use_select/' + str(land_use_name), 'r', encoding='UTF-8') as f:
        ind = json.load(fp=f)
    for key, values in ind.items():
        if (key in land_use_info_combine.keys()) and (len(values)>=1):
            for i in range(len(values)):
                land_use_info_combine[key].append(values[i])
        if (key not in land_use_info_combine.keys()) and (len(values)>=1):
            land_use_info_combine[key] = values

with open('./dataset/land_use_select/land_use_info_all.json', 'w') as f:
    json.dump(land_use_info_combine, f) #352

#%% 重新处理poi访问类别数量信息
cbsa_name_list_all = ['NY','GLA','WB','SFB','DFW', 'GB','GH','DV','AL', 'MM', 'MD', 'PS', 'GO', 'MSP', \
                  'DA', 'NO', 'PLM', 'GSL', 'CC', 'SM', 'PNC', 'IM', 'RT','GM','HR']
poi_class_set_dict = {}
for name in cbsa_name_list_all:
    dataset_name = "Weeplace_"+str(name)
    with open('./dataset/'+str(dataset_name)+'/user_travel_in_'+str(name)+'.json', 'r',encoding='UTF-8') as f:
        user_in_MSA_final = json.load(fp=f)
    for key, value in user_in_MSA_final.items(): #统计poi类别信息
        for i in range(1, len(value)):
            # o_county_id = str(value[i - 1][4]) + str(value[i - 1][5]) + str(value[i - 1][7])  # +str(value[i-1][8])
            d_county_id = str(value[i][4]) + str(value[i][5]) + str(value[i][7])  # +str(value[i][8])
            d_poi_class = str(value[i][2])
            if d_county_id not in poi_class_set_dict.keys():
                poi_class_set_dict[d_county_id] = []
            else:
                poi_class_set_dict[d_county_id].append(d_poi_class)

poi_class_num_dict = {}
for key,values in poi_class_set_dict.items():
    poi_class_set_dict[key] = list(set(values))
    poi_class_num_dict[key] = len(poi_class_set_dict[key])

with open('./dataset/poi_class_num_dict.json', 'w') as f:
    json.dump(poi_class_num_dict, f) #18271

#%% 拼接原始出行信息和土地 道路信息
cbsa_name_list_all = ['NY','GLA','WB','SFB','DFW', 'GB','GH','DV','AL', 'MM', 'MD', 'PS', 'GO', 'MSP', \
                  'DA', 'NO', 'PLM', 'GSL', 'CC', 'SM', 'PNC', 'IM', 'RT','GM','HR']
cbsa_name_list = ['NY','GLA','WB','SFB','DFW', 'GB','GH','DV','AL', 'MM', 'MD', 'PS']
with open('./dataset/road_density_set.json', 'r', encoding='UTF-8') as f:
    road_density_set = json.load(fp=f)
with open('./dataset/land_use_select/land_use_info_all.json', 'r', encoding='UTF-8') as f:
    land_use_set = json.load(fp=f)
with open('./dataset/poi_class_num_dict.json', 'r', encoding='UTF-8') as f:
    poi_class_num_dict = json.load(fp=f)

from collections import Counter
tract_info_all_combine = {}
for name in cbsa_name_list:
    with open('./dataset/region_results/region_all_info_set_dict_'+str(name)+'.json', 'r', encoding='UTF-8') as f:
        ind = json.load(fp=f)
        #seg value\o_flow_log\degree\level_num\ave_dis\poi_class_num\Entropy\Cluster_eff\pop\income
    for key, values in ind.items():
        if key in poi_class_num_dict.keys():
            values[5] = poi_class_num_dict[key]
        if key not in road_density_set.keys():
            continue
        if key not in tract_info_all_combine.keys():
            tract_info_all_combine[key] = values
            tract_info_all_combine[key].append(road_density_set[key][2])
            if key in land_use_set.keys():
                land_use_class_set = Counter(land_use_set[key])
                ss = 0
                for subkey,va in land_use_class_set.items():
                    land_use_class_set[subkey] = va/len(land_use_set[key])
                    ss += -land_use_class_set[subkey] * np.log(land_use_class_set[subkey])
                tract_info_all_combine[key].append(ss)
            else:
                tract_info_all_combine[key].append(0)

with open('./dataset/tract_info_all_combine2.json', 'w') as f:
    json.dump(tract_info_all_combine, f) #18271

#%%
with open('./dataset/tract_info_all_combine2.json', 'r', encoding='UTF-8') as f:
    tract_info_all_combine = json.load(fp=f)

cbsa_name_list_all = ['NY','GLA','WB','SFB','DFW', 'GB','GH','DV','AL', 'MM', 'MD', 'PS', 'GO', 'MSP', \
                      'DA', 'NO', 'PLM', 'GSL', 'CC', 'SM', 'PNC', 'IM', 'RT','GM','HR']

with open('./dataset/road_density_set.json', 'r', encoding='UTF-8') as f:
    road_density_set = json.load(fp=f)
with open('./dataset/land_use_select/land_use_info_all.json', 'r', encoding='UTF-8') as f:
    land_use_set = json.load(fp=f)

from collections import Counter
land_region,income, road_region = [], [],[]
for region in cbsa_name_list_all:
    dataset_name = 'Weeplace_' + region
    cbsa_set = pd.read_csv("./dataset/Region/" + str(region) + ".csv", header=None)  # cbsa region set
    # tract tess
    for i in range(len(cbsa_set)):
        if i == 0:
            tess_tract_selected = tess_tract_all[(tess_tract_all['STATENAME'] == cbsa_set.iloc[i][1]) & (tess_tract_all['COUNTYNAME'] == cbsa_set.iloc[i][0])]
            current_len = len(tess_tract_selected)
        else:
            tess_tract_selected = pd.concat((tess_tract_selected, tess_tract_all[(tess_tract_all['STATENAME'] == cbsa_set.iloc[i][1]) & (tess_tract_all['COUNTYNAME'] == cbsa_set.iloc[i][0])]))
            if len(tess_tract_selected) <= current_len:
                print(1)
                print(i)
            current_len = len(tess_tract_selected)
    # group tess
    for i in range(len(cbsa_set)):
        if i == 0:
            tess_group_selected = tess_all[(tess_all['STATENAME'] == cbsa_set.iloc[i][1]) & (tess_all['COUNTYNAME'] == cbsa_set.iloc[i][0])]
            current_len = len(tess_group_selected)
        else:
            tess_group_selected = pd.concat((tess_group_selected, tess_all[(tess_all['STATENAME'] == cbsa_set.iloc[i][1]) & (tess_all['COUNTYNAME'] == cbsa_set.iloc[i][0])]))
            if len(tess_group_selected) <= current_len:
                print(2)
                print(i)
            current_len = len(tess_group_selected)

    tess_tract_selected = tess_tract_selected.reset_index(drop=True);tess_tract_selected = tess_tract_selected.to_crs("epsg:4269");tess_tract_selected.crs
    tess_group_selected = tess_group_selected.reset_index(drop=True);tess_group_selected = tess_group_selected.to_crs("epsg:4269");tess_group_selected.crs

    #计算平均收入
    income_sum, income_count = 0, 0
    for i in range(len(tess_tract_selected)):
        geoid = tess_tract_selected.iloc[i].GEOID
        tess_ind = tess_group_selected[tess_group_selected['index'].str.startswith(geoid)]
        income_ = tess_ind['INCOME'].sum()
        # if income_>0:
        income_sum += income_
        income_count += 1
    income_ave = income_sum/income_count
    road_density_sum,road_density_count = 0, 0
    land_sum,land_count = 0, 0
    for i in range(len(tess_tract_selected)):
        key = str(tess_tract_selected.iloc[i].STATEFP)+str(tess_tract_selected.iloc[i].COUNTYFP)+str(tess_tract_selected.iloc[i].TRACTCE)
        if key in road_density_set.keys():
            road_density_sum += road_density_set[key][2]
            road_density_count += 1
        else:
            road_density_count += 1
        if key in land_use_set.keys():
            land_use_class_set = Counter(land_use_set[key])
            ss = 0
            for subkey,va in land_use_class_set.items():
                land_use_class_set[subkey] = va/len(land_use_set[key])
                ss += -land_use_class_set[subkey] * np.log(land_use_class_set[subkey])
            land_sum += ss; land_count+=1
        else:
            land_count+=1

    land_ave = land_sum/land_count
    road_ave = road_density_sum/road_density_count
    land_region.append(land_ave)
    road_region.append(road_ave)
    income.append(income_ave)

land_region = np.array(land_region).reshape(-1,1)
road_region = np.array(road_region).reshape(-1,1)
income = np.array(income).reshape(-1,1)



















