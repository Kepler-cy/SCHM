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
county_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/tl_2013_us_county.shp', encoding = 'gb18030')
tess_tract_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/TRACT2010.shp', encoding = 'gb18030')
tess_all = gpd.GeoDataFrame.from_file('./dataset/mapdata/data_2010.shp', encoding = 'gb18030')
cbsa_set = pd.read_csv("./dataset/NY_NJ_PA_BSA.csv", header=None) #cbsa region set
#county tess
for i in range(len(cbsa_set)):
    if i==0:
        tess_county_all_selected = county_all[(county_all['STATEFP']==str(cbsa_set.iloc[i][2])) & (county_all['NAMELSAD']==cbsa_set.iloc[i][0])]
    else:
        tess_county_all_selected = pd.concat((tess_county_all_selected,county_all[(county_all['STATEFP']==str(cbsa_set.iloc[i][2])) & (county_all['NAMELSAD']==cbsa_set.iloc[i][0])]))
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

tess_county_all_selected = tess_county_all_selected.reset_index(drop=True); tess_county_all_selected=tess_county_all_selected.to_crs("epsg:4269"); tess_county_all_selected.crs
tess_tract_selected = tess_tract_selected.reset_index(drop=True); tess_tract_selected=tess_tract_selected.to_crs("epsg:4269"); tess_tract_selected.crs
tess_group_selected = tess_group_selected.reset_index(drop=True); tess_group_selected=tess_group_selected.to_crs("epsg:4269"); tess_group_selected.crs

#%% 加载数据
dataset_all = pd.read_csv("./dataset/weeplace_checkins.csv",header=0) #7658368
dataset_all.columns=['uid','placeid','time','lat','lng','city','poi']

#%% 获取每个个体的home location，基于 21:00-6:00\\\只关注在目标区域有活动的个体，尽管他们的家不一定在目标区域
tess_group_selected=tess_group_selected.to_crs("epsg:4269") #原始4269
xmin, ymin, xmax, ymax = tess_group_selected.total_bounds
dataset1 = dataset_all[(dataset_all['lng']>=xmin) & (dataset_all['lng']<=xmax)]
dataset1 = dataset1[(dataset1['lat']>=ymin) & (dataset1['lat']<=ymax)] #1201173
u_id_list =set(dataset1['uid'].values)
dataset = dataset_all[dataset_all["uid"].isin(u_id_list)] #3442022

#获取每个个体的home location，基于home、dorm等信息
u_home_info = {}
for i in range(len(dataset)):
    hour = int(datetime.datetime.strptime(dataset.iloc[i][2], '%Y-%m-%dT%H:%M:%S').strftime('%H'))
    if (hour>=21 and hour<=24) or (hour>=0 and hour<=6):
        # if "Other:Home" in str(dataset.iloc[i][6]) or \
        #         "Other:Homes" in str(dataset.iloc[i][6]) or \
        #         "Dorm" in str(dataset.iloc[i][6]) or \
        #         "Others:Apartment Buildings" in str(dataset.iloc[i][6]) or \
        #         "Homes, Work, Others" in str(dataset.iloc[i][6]):
        if str(dataset.iloc[i][0]) not in u_home_info.keys():
            u_home_info[str(dataset.iloc[i][0])]={}
            u_home_info[str(dataset.iloc[i][0])][dataset.iloc[i][6]] = 1
            continue
        if dataset.iloc[i][6] not in u_home_info[str(dataset.iloc[i][0])].keys():
            u_home_info[str(dataset.iloc[i][0])][dataset.iloc[i][6]] = 1
        else:
            u_home_info[str(dataset.iloc[i][0])][dataset.iloc[i][6]] +=1

with open('./dataset/'+str(dataset_name)+'/u_home_info.json', 'w') as f:
    json.dump(u_home_info, f)

u_home_info_ind, u_home_info_final = {}, {}
for keys,values in u_home_info.items():
    tot = 0
    for key, v in values.items():
        # if key != 'Home / Work / Other:Home' and key != 'Home / Work / Other:Homes' and key!='College & Education:Dorm' and \
        #         key!='Homes, Work, Others:Homes' and key!='Colleges & Universities:College Dorms':
        #     continue
        if key is np.nan:
            continue
        if v > tot:
            tot = v
            u_home_info_ind[keys] = key
    if keys in u_home_info_ind.keys():
        u_home_info_final[keys] = []
        u_home_info_final[keys].append(str(u_home_info_ind[keys]))
        u_home_info_final[keys].append(dataset[(dataset['uid']==keys) & (dataset['poi']==str(u_home_info_ind[keys]))]['lat'].value_counts().idxmax())
        u_home_info_final[keys].append(dataset[(dataset['uid']==keys) & (dataset['poi']==str(u_home_info_ind[keys]))]['lng'].value_counts().idxmax())

with open('./dataset/'+str(dataset_name)+'/u_home_info_final.json', 'w') as f: #total 3163 users
    json.dump(u_home_info_final, f)

#%%[用tess id 信息来表示所有区域所有个体的home信息 2677】
u_home_info_final={}
with open('./dataset/'+str(dataset_name)+'/u_home_info_final.json', 'r',encoding='UTF-8') as f:
    u_home_info_final= json.load(fp=f) #5600

start_time = datetime.datetime.now()
user_home_info_all_final={}
for key, values in u_home_info_final.items():
    bound_county = county_all["geometry"]
    county_num = -1
    county_name = None
    ind=[]
    for index, bound in enumerate(bound_county):
        if bound.contains(geometry.Point(values[2], values[1])):
            ind.append(values[1]) #lat
            ind.append(values[2]) #lng
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
        if bound.contains(geometry.Point(values[2], values[1])):
            tract_num = str(bound_tract.iloc[index].TRACTCE)
            ind.append(tract_num)  # County id
            break
    if tract_num == "":
        continue
    block_num = -1
    bound_group = copy.deepcopy(tess_all[(tess_all['COUNTYFP']==county_num) & (tess_all['TRACTCE']==tract_num)])
    bound_group_geom = bound_group['geometry']
    if len(bound_group_geom)==0:
        break
    for index, bound in enumerate(bound_group_geom):
        if bound.contains(geometry.Point(values[2], values[1])):
            block_num = index
            ind.append(str(bound_group.iloc[index].BLKGRPCE))
            break
    if block_num == -1:
        continue
    group_info = str(ind[2])+str(ind[3])+str(ind[5])+str(ind[6])
    user_home_info_all_final[key]=group_info

# 保存用户home的地理信息，精细化到group level
with open('./dataset/'+str(dataset_name)+'/user_home_info_all_final.json', 'w') as f:
    json.dump(user_home_info_all_final, f)

u_info = {}
for i in range(len(dataset)):
    if str(dataset.iloc[i][0]) in u_home_info_final.keys(): #需要保证个体有home info
        if str(dataset.iloc[i][0]) not in u_info.keys():
            u_info[str(dataset.iloc[i][0])] = []
        ind = []
        ind.append(parser.parse(str(dataset.iloc[i][2]))) #时间
        ind.append(dataset.iloc[i][3]) #纬度
        ind.append(dataset.iloc[i][4]) #经度
        ind.append(dataset.iloc[i][6]) #poi
        ind.append(datetime.datetime.strptime(dataset.iloc[i][2], '%Y-%m-%dT%H:%M:%S').strftime('%H'))
        u_info[str(dataset.iloc[i][0])].append(ind)

#过滤数据 去除连续的数据、地点挨着的数据、ind出行次数需要>8次， 保证其代表性
u_id ={}
u_info_process = {}
for key,values in u_info.items():
    if key not in user_home_info_all_final.keys():
        continue
    v = copy.deepcopy(values)
    v.sort() #对出行记录进行时间先后顺序排列
    ind=[]
    ind.append(v[0])
    for i in range(1,len(v)):
        diff_t = (v[i][0]-v[i-1][0]).seconds
        diff_dis = utils.earth_distance([v[i][2],v[i][1]],[v[i-1][2],v[i-1][1]])
        if diff_t<3600 or diff_dis<0.1: 
            continue
        else:
            ind.append(v[i])
    if len(ind)>=8: #个体出行次数要大于8次
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
        u_info_process1[key].append(value[j][1:]) #因为time数据存不下来

with open('./dataset/'+str(dataset_name)+'/travel_data_info_process_all_regions.json', 'w') as f:
    json.dump(u_info_process1, f)
u_info_process={}
with open('./dataset/'+str(dataset_name)+'/travel_data_info_process_all_regions.json', 'r',encoding='UTF-8') as f:
    u_info_process = json.load(fp=f) #2675/4700
user_home_info_all_final={}
with open('./dataset/'+str(dataset_name)+'/user_home_info_all_final.json', 'r',encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f) #2677/4715

start_time = datetime.datetime.now()
user_in_all_region={}
for key, values in u_info_process.items():
    user_in_all_region[key]=[]
    for i in range(len(values)):
        #提取各个城市的边界
        bound_county = county_all["geometry"]
        county_num = -1
        county_name = None
        ind=[]
        for index, bound in enumerate(bound_county):
            if bound.contains(geometry.Point(values[i][1], values[i][0])):
                ind.append(values[i][0]) #lat
                ind.append(values[i][1]) #lng
                ind.append(values[i][2]) #poi
                ind.append(values[i][3]) #time
                ind.append(str(county_all.iloc[index].STATEFP)) #state id
                ind.append(str(county_all.iloc[index].COUNTYFP)) #County id
                ind.append(str(county_all.iloc[index].NAMELSAD)) #County name
                county_num = str(county_all.iloc[index].COUNTYFP)
                county_name = str(county_all.iloc[index].NAMELSAD)
                break
        if county_num == -1:
            continue
        #基于county id, 定位相应CBG的地理边界信息，数据对应的是 tess_county_selected
        tract_num = ""
        bound_tract = copy.deepcopy(tess_tract_all[tess_tract_all['COUNTYFP']==county_num])
        bound_tract_geom = bound_tract['geometry']
        for index, bound in enumerate(bound_tract_geom):
            if bound.contains(geometry.Point(values[i][1], values[i][0])):
                tract_num = str(bound_tract.iloc[index].TRACTCE)
                ind.append(tract_num)  # County id
                break
        if tract_num == "":
            continue
        block_num = -1
        bound_group = copy.deepcopy(tess_all[(tess_all['COUNTYFP']==county_num) & (tess_all['TRACTCE']==tract_num)])
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
        user_in_all_region[key].append(ind)

end_time = datetime.datetime.now()
print("数据分配时间："+str((end_time-start_time).seconds)+"秒") #数据分配时间：421s
user_in_all_region_final= {}
k=0
for key,value in user_in_all_region.items():
    if len(value)<8:
        continue
    k+=len(value)
    user_in_all_region_final[key]=value

with open('./dataset/'+str(dataset_name)+'/user_in_all_region_final.json', 'w') as f:
    json.dump(user_in_all_region_final, f)

#%%可用用户的income分析，计算income level分位数
user_home_info_all_final={}
with open('./dataset/'+str(dataset_name)+'/user_home_info_all_final.json', 'r',encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f) #4715

user_income = []
cencus_income = []
for key, values in user_home_info_all_final.items():
    tess_ind = tess_all[tess_all['index'].str.startswith(values[:-1])]
    income_sum = tess_ind['INCOME'].mean()
    v = income_sum
    if v>=0:
        user_income.append(v)

for i in range(len(tess_tract_selected)):
    v = tess_tract_selected.iloc[i].INCOME
    if v>=0:
        cencus_income.append(v)

res = np.percentile(user_income, (25, 50, 75), interpolation='midpoint') # [87155., 141427., 239047.]
res2 = np.percentile(cencus_income, (25, 50, 75), interpolation='midpoint') # [ 52727.5  87142.  142715. ]
#计算数据覆盖率
k1,k2,k3,k4=0,0,0,0
for i in range(len(user_income)):
    if user_income[i]<=res[0] and user_income[i]<=res2[0]:
        k1+=1
        continue
    if user_income[i]<=res[1] and user_income[i]<=res2[1]:
        k2+=1
        continue
    if user_income[i]<=res[2] and user_income[i]<=res2[2]:
        k3+=1
        continue
    if user_income[i] > res[2] and user_income[i] > res2[2]:
        k4 += 1
        continue

# #数据覆盖率
# res[1]-res[0] == res[2]-res[1]
# k1+k2+k3+k4
# 3468/4610= 0.75

user_income = np.array(user_income).T
cencus_income = np.array(cencus_income).T
x = np.arange(len(user_income))
plt.plot(x, user_income, 'o--')
plt.show()

#%% 计算所有 location 的 segregation level
q_level_cut_point =  [87155., 141427., 239047.] 
user_home_info_all_final = {}
with open('./dataset/'+str(dataset_name)+'/user_home_info_all_final.json', 'r',encoding='UTF-8') as f:
    user_home_info_all_final = json.load(fp=f) #7922

user_in_all_region_final = {}
with open('./dataset/'+str(dataset_name)+'/user_in_all_region_final.json', 'r',encoding='UTF-8') as f:
    user_in_all_region_final = json.load(fp=f) #

d_in_num = {} #
for key,value in user_in_all_region_final.items():
    o_county_id = user_home_info_all_final[key] #用户的home id
    for i in range(len(value)):
        d_county_id = str(value[i][4])+str(value[i][5])+str(value[i][7])
        if d_county_id not in d_in_num.keys():
            d_in_num[d_county_id] = {}
        if o_county_id not in d_in_num[d_county_id].keys():
            d_in_num[d_county_id][o_county_id] = 1

#计算user_home_info_all_final中各个home 对应的income
user_home_info_all_final_income = {}
for key,values in user_home_info_all_final.items():
    tess_ind = tess_all[tess_all['index'].str.startswith(values[:-1])]
    income_sum = tess_ind['INCOME'].mean()
    user_home_info_all_final_income[key] = income_sum

q1,q2,q3,q4 = 0,0,0,0
ttt=0
location_segregation_level = {} #13156
for key, value in user_in_all_region_final.items(): 
    user_ind_income = user_home_info_all_final_income[key] 
    # o_county_id = user_home_info_all_final[key]
    for i in range(len(value)):
        d_county_id = str(value[i][4])+str(value[i][5])+str(value[i][7])#+str(value[i][8])
        if len(d_in_num[d_county_id]) < 2:  
            ttt+=1
            continue
        if d_county_id not in location_segregation_level.keys():
            location_segregation_level[d_county_id] = np.zeros(4)
        if user_ind_income <= q_level_cut_point[0]:
            location_segregation_level[d_county_id][0] += 1; q1 += 1
        elif user_ind_income <= q_level_cut_point[1]:
            location_segregation_level[d_county_id][1] += 1; q2 += 1
        elif user_ind_income <= q_level_cut_point[2]:
            location_segregation_level[d_county_id][2] += 1; q3 += 1
        else:
            location_segregation_level[d_county_id][3] += 1; q4 += 1

aa, bb = [q1,q2,q3,q4],[q1,q2,q3,q4]
aa=aa/np.sum(bb) #[0.28545403 0.25083536 0.25891962 0.204791  ]
for key,values in location_segregation_level.items():
    sum_ = np.sum(values)
    if sum_!=0:
        location_segregation_level[key] = values/sum_

for key,values in location_segregation_level.items():
    location_segregation_level[key] = values.tolist()

with open('./dataset/'+str(dataset_name)+'/All_locations_segregation_level-样本1.json', 'w') as f: #2221-  2429中的部分位置的出行部分出行来自2个地点
    json.dump(location_segregation_level, f)

location_segregation_level={}
with open('./dataset/'+str(dataset_name)+'/All_locations_segregation_level-样本1.json', 'r',encoding='UTF-8') as f:
    location_segregation_level = json.load(fp=f)

location_segregation_value = {}
for key, values in location_segregation_level.items(): 
    sum_ = 0.0
    for i in range(len(values)):
        sum_ += (2.0/3 * abs(values[i]-0.25))
    location_segregation_value[key] = 1 - sum_ 
with open('./dataset/'+str(dataset_name)+'/All_locations_segregation_value1.json', 'w') as f: #2221
    json.dump(location_segregation_value, f)


location_segregation_level2={}
with open('./dataset/'+str(dataset_name)+'/All_locations_segregation_level-总体.json', 'r',encoding='UTF-8') as f:
    location_segregation_level2 = json.load(fp=f)

location_segregation_value2 = {}
for key, values in location_segregation_level2.items():
    sum_ = 0.0
    for i in range(len(values)):
        sum_ += (2.0/3 * abs(values[i]-0.25))
    location_segregation_value2[key] = 1 - sum_

seg_v1,seg_v2 = [],[]
for key, values in location_segregation_value2.items():
    seg_v1.append(location_segregation_value[key])
    seg_v2.append(location_segregation_value2[key])

def draw_scatter(data1,data2,x_name,y_name,fig_name):
    font1 = {'family': 'Arial', 'color': 'Black', 'size': 16}
    figsize = 6, 6
    plt.rcParams['xtick.direction'] = 'in'  
    plt.rcParams['ytick.direction'] = 'in' 
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='x', length=4, width=1, which='minor', top='on')  # ,top=True
    ax.tick_params(axis='x', length=8, width=1, which='major', top='on')  # ,right=True
    ax.tick_params(axis='y', length=8, width=1, which='major', right='on')  # ,top=True
    ax.tick_params(axis='y', length=4, width=1, which='minor', right='on')  # ,right=True
    ax.spines['bottom'].set_linewidth(1);  
    ax.spines['left'].set_linewidth(1);  
    ax.spines['right'].set_linewidth(1); 
    ax.spines['top'].set_linewidth(1);
    import seaborn as sns
    sns.kdeplot(data1, data2, cmap='Blues', fill=True)
    plt.xlabel(x_name, fontdict=font1)
    plt.ylabel(y_name, fontdict=font1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("./Figure/"+str(fig_name)+".png", dpi=360, bbox_inches='tight')

# draw_scatter(seg_v1,seg_v2,"cencus_seg","block_seg",'cencus-blocks-seg')
draw_scatter(seg_v1,seg_v2,"cencus_seg","block_seg",'cencus-seg')
from scipy import stats
pcc = stats.pearsonr(np.array(seg_v1).reshape(-1),np.array(seg_v2).reshape(-1)) #0.65



