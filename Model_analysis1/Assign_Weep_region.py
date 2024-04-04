import pandas as pd
import numpy as np
import json
import geopandas as gpd
from shapely import geometry
from dateutil import parser
import copy
import datetime
import utils
import draw_utils
import matplotlib.pyplot as plt
region = 'NY_NJ_PA_BSA'
dataset_name = 'Weeplace'

#%% [基于bsa筛选区域，county、tract、group]
print(1)
# county_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_county.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding = 'gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding = 'gb18030')
cbsa_set = pd.read_csv("./dataset/NY_NJ_PA_BSA.csv",header=None) #cbsa region set
#county tess
# for i in range(len(cbsa_set)):
#     if i==0:
#         tess_county_all_selected = county_all[(county_all['STATEFP']==str(cbsa_set.iloc[i][2])) & (county_all['NAMELSAD']==cbsa_set.iloc[i][0])]
#     else:
#         tess_county_all_selected = pd.concat((tess_county_all_selected,county_all[(county_all['STATEFP']==str(cbsa_set.iloc[i][2])) & (county_all['NAMELSAD']==cbsa_set.iloc[i][0])]))
#tract tess
for i in range(len(cbsa_set)):
    if i==0:
        tess_tract_selected = tess_tract_all[(tess_tract_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_tract_all['COUNTYNAME']==cbsa_set.iloc[i][0])]
    else:
        tess_tract_selected = pd.concat((tess_tract_selected,tess_tract_all[(tess_tract_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_tract_all['COUNTYNAME']==cbsa_set.iloc[i][0])]))
#group tess
for i in range(len(cbsa_set)):
    if i==0:
        tess_group_selected = tess_all[(tess_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_all['COUNTYNAME']==cbsa_set.iloc[i][0])]
    else:
        tess_group_selected = pd.concat((tess_group_selected,tess_all[(tess_all['STATENAME']==cbsa_set.iloc[i][1]) & (tess_all['COUNTYNAME']==cbsa_set.iloc[i][0])]))

tess_tract_income = []
tess_tract_pop = []
for i in range(len(tess_tract_selected)):
    geoid = tess_tract_selected.iloc[i].GEOID
    tess_ind = tess_group_selected[tess_group_selected['index'].str.startswith(geoid)]
    income_sum = tess_ind['INCOME'].mean()
    pop_sum = tess_ind['POPULATION'].mean()
    tess_tract_income.append(income_sum)
    tess_tract_pop.append(pop_sum)
tess_tract_selected['INCOME'] = tess_tract_income
tess_tract_selected['POPULATION'] = tess_tract_pop

# tess_county_all_selected = tess_county_all_selected.reset_index(drop=True); tess_county_all_selected=tess_county_all_selected.to_crs("epsg:4269"); tess_county_all_selected.crs
tess_tract_selected = tess_tract_selected.reset_index(drop=True); tess_tract_selected=tess_tract_selected.to_crs("epsg:4269"); tess_tract_selected.crs
tess_group_selected = tess_group_selected.reset_index(drop=True); tess_group_selected=tess_group_selected.to_crs("epsg:4269"); tess_group_selected.crs

#%% 加载数据
dataset_all = pd.read_csv("./dataset/weeplace_checkins.csv",header=0) #7658368
dataset_all.columns=['uid','placeid','time','lat','lng','city','poi']

start_time = datetime.datetime.now()
tess_group_selected=tess_group_selected.to_crs("epsg:4269") #
xmin, ymin, xmax, ymax = tess_group_selected.total_bounds
dataset = dataset_all[(dataset_all['lng']>=xmin) & (dataset_all['lng']<=xmax)]
dataset = dataset[(dataset['lat']>=ymin) & (dataset['lat']<=ymax)] #220642

u_info = {}
for i in range(len(dataset)):
    if str(dataset.iloc[i][0]) not in u_info.keys():
        u_info[str(dataset.iloc[i][0])] = []
    ind = []
    ind.append(parser.parse(str(dataset.iloc[i][2]))) #
    ind.append(dataset.iloc[i][3]) #
    ind.append(dataset.iloc[i][4]) #
    ind.append(dataset.iloc[i][6]) #poi
    ind.append(datetime.datetime.strptime(dataset.iloc[i][2], '%Y-%m-%dT%H:%M:%S').strftime('%H'))
    u_info[str(dataset.iloc[i][0])].append(ind)

u_id ={}
u_info_process = {}
for key,values in u_info.items():
    v = copy.deepcopy(values)
    v.sort() #
    ind=[]
    ind.append(v[0])
    for i in range(1,len(v)):
        diff_t = (v[i][0]-v[i-1][0]).seconds
        diff_dis = utils._distance([v[i][2],v[i][1]],[v[i-1][2],v[i-1][1]])
        if diff_t<3600 or diff_dis<0.1: # 
            continue
        else:
            ind.append(v[i])
    if len(ind)>=8: #
        u_id[key]=len(ind)
        u_info_process[key] = ind

end_time = datetime.datetime.now()
print("数据过滤时间："+str((end_time-start_time).seconds)+"秒")
u_info_process1={}
sum_=0
for key, value in u_info_process.items():
    u_info_process1[key]=[]
    sum_+=len(value)
    for j in range(len(value)):
        value[j][3] = str(value[j][3])
        u_info_process1[key].append(value[j][1:]) #

with open('./dataset/'+str(dataset_name)+'/travel_data_in_region_'+str(region)+'.json', 'w') as f:
    json.dump(u_info_process1, f)

u_info_process={}
with open('./dataset/'+str(dataset_name)+'/travel_data_in_region_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    u_info_process = json.load(fp=f)
user_home_info_all_final={}
with open('./dataset/'+str(dataset_name)+'/user_home_info_all_final.json', 'r',encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f) #2677

start_time = datetime.datetime.now()
user_in_MSA={}
for key, values in u_info_process.items():
    if key not in user_home_info_all_final.keys(): 
        continue
    user_in_MSA[key]=[]
    for i in range(len(values)):
        bound_county = tess_county_all_selected["geometry"]
        county_num = -1
        county_name = None
        ind=[]
        for index, bound in enumerate(bound_county):
            if bound.contains(geometry.Point(values[i][1], values[i][0])):
                ind.append(values[i][0]) #lat
                ind.append(values[i][1]) #lng
                ind.append(values[i][2]) #poi
                ind.append(values[i][3]) #time
                ind.append(str(tess_county_all_selected.iloc[index].STATEFP)) #state id
                ind.append(str(tess_county_all_selected.iloc[index].COUNTYFP)) #County id
                ind.append(str(tess_county_all_selected.iloc[index].NAMELSAD)) #County name
                county_num = str(tess_county_all_selected.iloc[index].COUNTYFP)
                county_name = str(tess_county_all_selected.iloc[index].NAMELSAD)
                break
        if county_num == -1:
            continue
        tract_num = ""
        bound_tract = copy.deepcopy(tess_tract_selected[tess_tract_selected['COUNTYFP']==county_num])
        bound_tract_geom = bound_tract['geometry']
        for index, bound in enumerate(bound_tract_geom):
            if bound.contains(geometry.Point(values[i][1], values[i][0])):
                tract_num = str(bound_tract.iloc[index].TRACTCE)
                ind.append(tract_num)  # County id
                break
        if tract_num == "":
            continue
        block_num = -1
        bound_group = copy.deepcopy(tess_group_selected[(tess_group_selected['COUNTYFP']==county_num) & (tess_group_selected['TRACTCE']==tract_num)])
        bound_group_geom = bound_group['geometry']
        if len(bound_group_geom)==0:
            break
        for index, bound in enumerate(bound_group_geom):
            if bound.contains(geometry.Point(values[i][1], values[i][0])):
                block_num = index
                ind.append(str(bound_group.iloc[index].BLKGRPCE))
                break
        if block_num == -1:
            continue
        ind.append(user_home_info_all_final[key])
        user_in_MSA[key].append(ind)

end_time = datetime.datetime.now()
print("数据分配时间："+str((end_time-start_time).seconds)+"秒") #数据分配时间：421s
user_in_MSA_final= {}
k=0
for key,value in user_in_MSA.items():
    if len(value)<8:
        continue
    k+=len(value)
    user_in_MSA_final[key]=value

with open('./dataset/'+str(dataset_name)+'/user_travel_in_'+str(region)+'.json', 'w') as f:
    json.dump(user_in_MSA_final, f)

user_in_MSA_final={}
with open('./dataset/'+str(dataset_name)+'/user_travel_in_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    user_in_MSA_final = json.load(fp=f)

odflow_msa = {}
msa_set = {}
for key, value in user_in_MSA_final.items():
    for i in range(1,len(value)):
        o_county_id = str(value[i-1][4])+str(value[i-1][5])+str(value[i-1][7])#+str(value[i-1][8])
        d_county_id = str(value[i][4])+str(value[i][5])+str(value[i][7])#+str(value[i][8])
        if (str(o_county_id), str(d_county_id)) not in odflow_msa.keys():
            odflow_msa[(str(o_county_id), str(d_county_id))] = 1
        else:
            odflow_msa[(str(o_county_id), str(d_county_id))] += 1
        msa_set[str(o_county_id)] = 1
        msa_set[str(d_county_id)] = 1

o2d2flow_msa = {}
for (o, d),f in odflow_msa.items():
    try:
        d2f = o2d2flow_msa[o]
        d2f[d] = f
    except KeyError:
        o2d2flow_msa[o] = {d: f}

with open('./dataset/'+str(dataset_name)+'/o2d2flow_msa_in_region_'+str(region)+'.json', 'w') as f: #3467
    json.dump(o2d2flow_msa, f)

tess_selected_final = tess_tract_selected[tess_tract_selected['GEOID'] == '36047000100']
tess_ind = {}
tess_ind['36047000100'] = 1
for key, values in o2d2flow_msa.items():
    if key not in tess_ind.keys():
        tess_ind[key] = 1
        tess_selected_final = pd.concat((tess_selected_final, tess_tract_selected[tess_tract_selected['GEOID'] == key]))
    for sub_key, va in values.items():
        if sub_key not in tess_ind.keys():
            tess_ind[sub_key] = 1
            tess_selected_final = pd.concat((tess_selected_final, tess_tract_selected[tess_tract_selected['GEOID'] == sub_key]))
tess_selected_final = tess_selected_final.reset_index(drop=True);
tess_selected_final = tess_selected_final.to_crs("epsg:4269"); tess_selected_final.crs


#%% 计算从各个location的被隔离值
user_home_info_all_final = {}
with open('./dataset/'+str(dataset_name)+'/user_home_info_all_final.json', 'r',encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f) #4715
user_in_MSA_final={}
with open('./dataset/'+str(dataset_name)+'/user_travel_in_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    user_in_MSA_final = json.load(fp=f) #3412
location_segregation_value={}
with open('./dataset/'+str(dataset_name)+'/All_locations_segregation_value1.json', 'r',encoding='UTF-8') as f:
    location_segregation_value = json.load(fp=f)  

location_segregation_select_value = {}  
for key,values in user_in_MSA_final.items():
    for i in range(len(values)):
        ind_d = str(values[i][4]) + str(values[i][5]) + str(values[i][7])
        if ind_d not in location_segregation_select_value.keys():
            if ind_d not in location_segregation_value.keys(): #5276
                continue
            location_segregation_select_value[ind_d] = location_segregation_value[ind_d]

with open('./dataset/'+str(dataset_name)+'/Location_segregation_value_in_region_'+str(region)+'1.json', 'w') as f:
    json.dump(location_segregation_select_value, f) #2641/3077

location_be_segregated_list = {}
for o_key, value in location_segregation_select_value.items():
    location_be_segregated_list[o_key] = []
    for user_name,values in user_in_MSA_final.items():
        seg_sum = 0.0
        k=0
        for i in range(len(values)-1): #user 出行链
            o_ind = str(values[i][4])+str(values[i][5])+str(values[i][7])
            if o_ind==o_key:
                d_id = str(values[i+1][4])+str(values[i+1][5])+str(values[i+1][7])
                if d_id in location_segregation_select_value.keys():
                    k+=1
                    seg_sum += location_segregation_select_value[d_id] #所有的d都是在目标区域内的
        if k!=0:
            location_be_segregated_list[o_key].append(seg_sum/k)

location_be_segregated_value = {}
for key,values in location_be_segregated_list.items():
    if len(values) == 0:
        continue
    else:
        location_be_segregated_value[key] = np.mean(values) #2641

with open('./dataset/'+str(dataset_name)+'/Location_be_segregated_value_in_region_'+str(region)+'1.json', 'w') as f: #2641
    json.dump(location_be_segregated_value, f)

user_in_MSA_final={}
with open('./dataset/'+str(dataset_name)+'/user_travel_in_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    user_in_MSA_final = json.load(fp=f) #3412
location_segregation_value={}
with open('./dataset/'+str(dataset_name)+'/Location_segregation_value_in_region_'+str(region)+'1.json', 'r',encoding='UTF-8') as f:
    location_segregation_value = json.load(fp=f) #2641
location_be_segregated_value = {}
with open('./dataset/' + str(dataset_name) + '/Location_be_segregated_value_in_region_'+str(region)+'1.json', 'r', encoding='UTF-8') as f:
    location_be_segregated_value = json.load(fp=f) #2641

odflow_msa = {}
mmax_ = -1
for key, value in user_in_MSA_final.items():
    for i in range(1,len(value)):
        o_county_id = str(value[i-1][4])+str(value[i-1][5])+str(value[i-1][7])
        d_county_id = str(value[i][4])+str(value[i][5])+str(value[i][7])
        if (str(o_county_id), str(d_county_id)) not in odflow_msa.keys():
            if (str(o_county_id) in location_be_segregated_value.keys()) and (str(d_county_id) in location_segregation_value.keys()):
                mmax_ = max(mmax_, location_be_segregated_value[o_county_id]*location_segregation_value[d_county_id])
                odflow_msa[(str(o_county_id), str(d_county_id))] = [1, location_be_segregated_value[o_county_id]*location_segregation_value[d_county_id]]
        else:
            odflow_msa[(str(o_county_id), str(d_county_id))][0] += 1

#转为能存储的形式
o2d2flow_msa = {}
for (o, d),f in odflow_msa.items():
    try:
        d2f = o2d2flow_msa[o]
        d2f[d] = f
    except KeyError:
        o2d2flow_msa[o] = {d: f}

import math
o2d2flow_msa_new = {}
for key, values in o2d2flow_msa.items():
    o2d2flow_msa_new[key] = {}
    sum_flow = 0
    for subkey,va in o2d2flow_msa[key].items():
        sum_flow += va[0]
    pop_o = tess_tract_selected[tess_tract_selected["GEOID"] == key].POPULATION.values[0]
    ratio = pop_o*1.0/sum_flow
    for subkey,va in o2d2flow_msa[key].items():
        ind = []
        ind.append(math.ceil(ratio*va[0])+1)
        ind.append(va[1]/mmax_)
        o2d2flow_msa_new[key][subkey] = ind

with open('./dataset/'+str(dataset_name)+'/o2d2flow_isi_msa_in_region_'+str(region)+'1.json', 'w') as f: #2272
    json.dump(o2d2flow_msa_new, f)

#%% 数据od分布、income、population可视化
o2d2flow_msa={}
with open('./dataset/'+str(dataset_name)+'/o2d2flow_isi_msa_in_region_'+str(region)+'1.json', 'r',encoding='UTF-8') as f:
    o2d2flow_msa = json.load(fp=f)

flowss = 0
group_od_flow = []
group_od_seg = []
for key,values in o2d2flow_msa.items():
    # o_lng = tess_tract_selected[tess_tract_selected['GEOID']==key].geometry.centroid.to_list()[0]
    for sub_key,value in values.items():
        if key ==sub_key or value[1]==0:
            continue
        group_od_flow.append(value[0])
        flowss+=1
        group_od_seg.append(value[1])

group_od_seg1= []
group_od_flow1= []
for i in range(len(group_od_flow)):
    group_od_flow1.append(group_od_flow[i])
    group_od_seg1.append(group_od_seg[i])

group_o_flow = {}
for key,values in o2d2flow_msa.items():
    group_o_flow[key] = 0
    for subkey, value in values.items():
        group_o_flow[key] += value[0]
    if group_o_flow[key]==0:
        group_o_flow[key] = 1
for i in range(len(tess_selected_final)):
    key_v = tess_selected_final.iloc[i].index.values[0]
    if key_v not in group_o_flow.keys():
        group_o_flow[key_v] = 1

group_d_flow = {}
for key,values in o2d2flow_msa.items():
    for subkey, value in values.items():
        if subkey not in group_d_flow.keys():
            group_d_flow[subkey] = 0
        group_d_flow[subkey] += value[0]
for i in range(len(tess_selected_final)):
    key_v = tess_selected_final.iloc[i].index.values[0]
    if key_v not in group_d_flow.keys():
        group_d_flow[key_v] = 1

group_o_flow_list = []
group_d_flow_list = []
group_seg_list = []
kk=0
for i in range(len(tess_selected_final)):
    if tess_selected_final.loc[i].INCOME <= 0:
        tess_selected_final['INCOME'][i] = 1
    if tess_selected_final.loc[i].POPULATION <= 0:
        tess_selected_final['POPULATION'][i] = 1
    if tess_selected_final.loc[i]['GEOID'] not in location_segregation_value.keys():
        group_o_flow_list.append(np.nan)
    elif tess_selected_final.loc[i]['GEOID'] not in group_o_flow.keys():
        group_o_flow_list.append(1)
    else:
        group_o_flow_list.append(group_o_flow[tess_selected_final.loc[i]['GEOID']])
    if tess_selected_final.loc[i]['GEOID'] not in location_segregation_value.keys():
        group_d_flow_list.append(np.nan)
    elif tess_selected_final.loc[i]['GEOID'] not in group_d_flow.keys():
        group_d_flow_list.append(1)
    else:
        group_d_flow_list.append(group_d_flow[tess_selected_final.loc[i]['GEOID']])
    if tess_selected_final.loc[i]['GEOID'] not in location_segregation_value.keys():
        group_seg_list.append(np.nan)
        kk+=1
    else:
        group_seg_list.append(location_segregation_value[tess_selected_final.loc[i]['GEOID']])

group_o_flow1 = np.log10(np.array(group_o_flow_list))
max_1 = np.max(group_o_flow1)
tess_selected_final['o_flow'] = group_o_flow_list
group_d_flow1 = np.log10(np.array(group_d_flow_list))
max_2 = np.max(group_d_flow1)
tess_selected_final['d_flow'] = group_d_flow_list
max_2 = np.max(group_seg_list)
tess_selected_final['seg_value'] = group_seg_list

block_tess = tess_selected_final.dropna(axis=0)
block_tess = block_tess.reset_index(drop=True);
block_tess = block_tess.to_crs("epsg:4269"); block_tess.crs

def draw_scatter(data1,data2,x_name,y_name,fig_name):
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
    ax.set_xlim(0, 1)
    # plt.scatter(data1,data2, cmap='jet', alpha=0.95, s=50, edgecolor='black')
    import seaborn as sns
    sns.kdeplot(data1, data2, cmap='Blues', fill=True)

    # 显示图像
    ax.set_ylim(0, 100)
    # ax.set_ylim(0, 20)
    plt.xlabel(x_name, fontdict=font1)
    plt.ylabel(y_name, fontdict=font1)
    plt.xticks(fontsize=16)  # 横轴刻度字体大小
    plt.yticks(fontsize=16)  # 纵轴刻度字体大小
    plt.savefig("./Figure/"+str(fig_name)+".png", dpi=360, bbox_inches='tight')

from scipy import stats
draw_scatter(block_tess['o_flow'].values,block_tess['seg_value'].values,"o_flow","group_seg",'outflow-seg')
pcc = stats.pearsonr(block_tess['o_flow'].values.reshape(-1),block_tess['seg_value'].values.reshape(-1)) #0.21
draw_scatter(block_tess['d_flow'].values,block_tess['seg_value'].values,"d_flow","group_seg",'inflow-seg')
pcc = stats.pearsonr(block_tess['d_flow'].values.reshape(-1),block_tess['seg_value'].values.reshape(-1))#0.21

aa= np.array(group_od_seg1).reshape(-1,1)
bb= np.array(group_od_flow1).reshape(-1,1)
draw_scatter(group_od_seg1,group_od_flow1,"od_seg","od_flow",'odflow-seg')
pcc = stats.pearsonr(group_od_seg1,group_od_flow1) #-0.201  #0.129
spc = stats.spearmanr(group_od_seg1,group_od_flow1) #-0.201  #0.246

import matplotlib as mpl
def draw2(tess,min_, max_,name,dataset_name,col_name):
    color_ = "Oranges"  #GnBu jet Greens  terrain  YlGn  Oranges
    font1 = {'family':'Arial','color': 'Black','size':16}
    #设置图片的大小
    figsize = 14,9
    li=0
    plt.rcParams['xtick.direction'] = 'in' ####坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in' ####坐标轴刻度朝内
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='x',length =4,width=li, which='minor',top='on') #,top=True
    ax.tick_params(axis='x',length =8, width=li, which='major',top='on')#,right=True
    ax.tick_params(axis='y',length =8,width=li, which='major', right='on') #,top=True
    ax.tick_params(axis='y',length =4,width=li, which='minor', right='on') #,right=True
    ax.spines['bottom'].set_linewidth(li);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(li);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(li);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(li);####设置上部坐标轴的粗细
    plt.xticks(fontsize=0) #横轴刻度字体大小
    plt.yticks(fontsize=0) #纵轴刻度字体大小
    tess.plot(ax=ax, edgecolor='gray',column=col_name, cmap=color_) #, cmap='jet'
    fig = figure
    cax = fig.add_axes([0.8, 0.27, 0.032, 0.15])
    if col_name=='seg_value':
        sm = mpl.cm.ScalarMappable(cmap=color_)
    else:
        # norm = mpl.colors.LogNorm(vmin=min_, vmax = max_)
        sm = mpl.cm.ScalarMappable(cmap=color_)
    sm._A = []
    bar = fig.colorbar(sm, cax=cax)
    bar.ax.tick_params(labelsize=14)  #设置色标刻度字体大小。
    plt.savefig("./Figure/Correlation_"+str(dataset_name)+"_"+str(name)+".png", dpi=360, bbox_inches = 'tight')

draw2(block_tess, 1, 1000, region,"o_flow","o_flow")
draw2(block_tess, 1, 5, region,"d_flow","d_flow")
draw2(block_tess, 1, 8000, region,"income","INCOME")
draw2(block_tess, 1, 2000, region,"population","POPULATION")
draw2(block_tess, 1, max_2, region,"seg_value","seg_value")#需要取消norm

#density fig of deg value
import seaborn as sns
def draw_distribution_data(dataset, numBins,region):
    ss=16; font1 = {'family' : 'Times new roman','color':'Black','size': ss+2}
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
    sh.set_xlabel("Tract segregation",fontdict=font1) #横轴名称
    sh.set_ylabel("density",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=ss) #横轴刻度字体大小
    plt.yticks(fontsize=ss) #纵轴刻度字体大小
    sns.histplot(dataset, bins = numBins, kde=True,alpha=0.45, color ='blue')
    plt.subplots_adjust(left=0,right=0.9)
    plt.savefig("./Figure/Tract segregation_"+str(region)+".png",dpi=360,bbox_inches = 'tight')
    # plt.show()
draw_distribution_data(group_seg_list, 30, region) #3112个tess

#%% 对出行数据进行#Shannon entropy
out_flow1,out_flow2,out_flow3,out_flow4,out_flow_all = [],[],[],[],[]
for i in range(len(block_tess)):
    seg_v = block_tess.iloc[i]['seg_value']
    block_name = block_tess.iloc[i]['GEOID']
    if block_name in o2d2flow_msa.keys():
        sum_ = 0
        for subkey,values in o2d2flow_msa[block_name].items():
            sum_ += values[0]
        ind = []
        for subkey,values in o2d2flow_msa[block_name].items():
            ind.append(values[0]/sum_)
        out_flow_all.append(ind)
        if seg_v<=0.25:
            out_flow1.append(ind)
        elif seg_v<=0.5:
            out_flow2.append(ind)
        elif seg_v<=0.75:
            out_flow3.append(ind)
        else:
            out_flow4.append(ind)

se_list1,se_list2,se_list3,se_list4,se_list_all = [],[],[],[],[]
for i in range(len(out_flow1)):
    se=0
    for j in range(len(out_flow1[i])):
        se += -1.0 * out_flow1[i][j] * np.log(out_flow1[i][j])
    se_list1.append(se)
for i in range(len(out_flow2)):
    se=0
    for j in range(len(out_flow2[i])):
        se += -1.0 * out_flow2[i][j] * np.log(out_flow2[i][j])
    se_list2.append(se)
for i in range(len(out_flow3)):
    se=0
    for j in range(len(out_flow3[i])):
        se += -1.0 * out_flow3[i][j] * np.log(out_flow3[i][j])
    se_list3.append(se)
for i in range(len(out_flow4)):
    se=0
    for j in range(len(out_flow4[i])):
        se += -1.0 * out_flow4[i][j] * np.log(out_flow4[i][j])
    se_list4.append(se)
for i in range(len(out_flow_all)):
    se=0
    for j in range(len(out_flow_all[i])):
        se += -1.0 * out_flow_all[i][j] * np.log(out_flow_all[i][j])
    se_list_all.append(se)

import numpy as np
import matplotlib.pyplot as plt
def get_cdf(data):
    data_fre =pd.Series(data).value_counts()
    data_fre_sort=data_fre.sort_index(axis=0,ascending=True)
    data_fre_sort_df=data_fre_sort.reset_index()
    data_fre_sort_df[0]=data_fre_sort_df[0]/len(data)#将频数转换成概率
    data_fre_sort_df.columns=['Rds','Fre']#将列表列索引重命名
    data_fre_sort_df['cumsum']=np.cumsum(data_fre_sort_df['Fre'])
    return data_fre_sort_df

se_list1_fre_sort_df = get_cdf(se_list1); se_list3_fre_sort_df = get_cdf(se_list3)
se_list2_fre_sort_df = get_cdf(se_list2); se_list4_fre_sort_df = get_cdf(se_list4)
se_list_all_fre_sort_df = get_cdf(se_list_all)

plot=plt.figure()
ax1=plot.add_subplot(1,1,1)
ax1.plot(se_list1_fre_sort_df['Rds'],se_list1_fre_sort_df['cumsum'],label="Seg level1")
ax1.plot(se_list2_fre_sort_df['Rds'],se_list2_fre_sort_df['cumsum'],label="Seg level2")
ax1.plot(se_list3_fre_sort_df['Rds'],se_list3_fre_sort_df['cumsum'],label="Seg level3")
ax1.plot(se_list4_fre_sort_df['Rds'],se_list4_fre_sort_df['cumsum'],label="Seg level4")
ax1.plot(se_list_all_fre_sort_df['Rds'],se_list_all_fre_sort_df['cumsum'],color='black',label="All")
plt.xlabel("Entropy $se$")
plt.ylabel("P(seg$_{i}$>$se$)")
plt.legend()
plt.savefig("./Figure/Entropy_of_different_seg_levels_" + str(region) + "_ISI.png", dpi=360, bbox_inches='tight')
plt.show()

#%% 对出行数据进行聚类系数分析
out_flow1,out_flow2,out_flow3,out_flow4,out_flow_all = [],[],[],[],[]
max_ = -1
min_ = 99999999
for i in range(len(block_tess)):
    block_name = block_tess.iloc[i]['GEOID']
    if block_name in o2d2flow_msa.keys():
        for subkey,values in o2d2flow_msa[block_name].items():
            max_ = max(max_,values[0])
            min_ = min(min_,values[0])
od_flow_new = {}
for i in range(len(block_tess)):
    block_name = block_tess.iloc[i]['GEOID']
    if block_name in o2d2flow_msa.keys():
        od_flow_new[block_name] = {}
        for subkey,values in o2d2flow_msa[block_name].items():
            # if values[0]>2:
            od_flow_new[block_name][subkey] = values[0]/max_#(values[0]-min_)/(max_-min_)

import math
c_list1, c_list2, c_list3, c_list4,c_all = [],[],[],[],[]
for i in range(len(block_tess)): #Ci
    seg_v = block_tess.iloc[i]['seg_value']
    block_name = block_tess.iloc[i]['GEOID']
    if block_name not in od_flow_new.keys():
        continue
    degree_i = len(od_flow_new[block_name])
    sum_ = 0
    for subkey_j, values_j in od_flow_new[block_name].items(): #遍历 j
        if subkey_j in od_flow_new.keys():
            for subkey_k, values_k in od_flow_new[subkey_j].items(): #遍历 k
                if subkey_k in od_flow_new.keys() and block_name in od_flow_new[subkey_k].keys():
                    sum_ += math.pow(od_flow_new[block_name][subkey_j]*od_flow_new[subkey_j][subkey_k]*od_flow_new[subkey_k][block_name], 1/3) #!
    if degree_i>1:
        ci = sum_ / (degree_i*(degree_i-1))
    c_all.append(ci)
    if seg_v<=0.25:
        c_list1.append(ci)
    elif seg_v<=0.5:
        c_list2.append(ci)
    elif seg_v<=0.75:
        c_list3.append(ci)
    else:
        c_list4.append(ci)

c_list1_fre_sort_df = get_cdf(c_list1); c_list3_fre_sort_df = get_cdf(c_list3)
c_list2_fre_sort_df = get_cdf(c_list2); c_list4_fre_sort_df = get_cdf(c_list4)
c_all_fre_sort_df = get_cdf(c_all)

plot=plt.figure()
ax1=plot.add_subplot(1,1,1)
ax1.plot(c_list1_fre_sort_df['Rds'],c_list1_fre_sort_df['cumsum'],label="Seg level1")
ax1.plot(c_list2_fre_sort_df['Rds'],c_list2_fre_sort_df['cumsum'],label="Seg level2")
ax1.plot(c_list3_fre_sort_df['Rds'],c_list3_fre_sort_df['cumsum'],label="Seg level3")
ax1.plot(c_list4_fre_sort_df['Rds'],c_list4_fre_sort_df['cumsum'],label="Seg level4")
ax1.plot(c_all_fre_sort_df['Rds'],c_all_fre_sort_df['cumsum'],color = 'black',label="All")
plt.xlabel("Clustering coefficient $c$")
plt.ylabel("P(Coef$_{i}$>$c$)")
ax1.set_xlim(0, 0.015)
plt.legend()
plt.savefig("./Figure/Clustering_coefficient_of_different_seg_levels_" + str(region) + "_ISI.png", dpi=360, bbox_inches='tight')
# plt.show()

