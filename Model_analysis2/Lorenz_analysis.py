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
cbsa_name_list_all = ['NY','GLA','WB','SFB', 'GB','DV','AL', 'MM', 'PS',  'MSP']

#%%
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

# %% 数据加载, 转化为od flow形式，并存储flow seg 和 loc name 至 value_flow_seg_name_list
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

# 存储flow\seg\loc name value
value_flow_seg_name_list = []
for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():  # 出发点约束在我们的研究区域内
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

value_flow_seg_name_list = sorted(value_flow_seg_name_list)  

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

value_flow_dict = {}
for i in range(len(value_flow_seg_name_list)):  
    if value_flow_seg_name_list[i][0] not in value_flow_dict.keys():  # seg\degree\flow——all\o_id
        value_flow_dict[value_flow_seg_name_list[i][0]] = []
        value_flow_dict[value_flow_seg_name_list[i][0]].append(value_flow_seg_name_list[i][0])  # 0：seg values
        value_flow_dict[value_flow_seg_name_list[i][0]].append(value_flow_seg_name_list[i][3])  # 3：oid
    else:
        value_flow_dict[value_flow_seg_name_list[i][0]][0] += value_flow_seg_name_list[i][0]
        value_flow_dict[value_flow_seg_name_list[i][0]].append(value_flow_seg_name_list[i][3])

value_seg_list_sort = []  
for key, values in value_flow_dict.items():
    value_seg_list_sort.append(values)
value_seg_list_sort = sorted(value_seg_list_sort)

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


def calculate_slope(x, y): #计算斜率
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    slope = dy / dx
    print(dy);
    print(dx);
    y_intercept = y[-1] - slope * x[-1]
    x_intersection = -y_intercept / slope
    return slope, x_intersection

value_seg_cir_level_all, value_rank_cir_level_all, loc_level_all, x_intersection, slopes = [], [], [], [],[]
value_seg_cir_level_all.append(value_flow_cir_level_1); value_rank_cir_level_all.append(value_rank_cir_level_1)
inter_pre = 1
for ii in range(20):
    slope1, inter_new = calculate_slope(value_rank_cir_level_all[ii], value_seg_cir_level_all[ii])
    print(f"Slope at level {ii}: {slope1}")
    print(f"x_intersection: {inter_new}")
    x_intersection.append(inter_new)
    slopes.append(slope1)
    ss_sum, k, tot = 0., 0., 0
    for i in range(len(value_seg_list_sort)):
        k += 1
        if k / len(value_seg_list_sort) <= inter_new: 
            ss_sum += value_seg_list_sort[i][0]; tot += 1

    ss, k = 0., 0.0
    value_seg_cir_level_ind, value_rank_cir_level_ind, loc_level_ind = [], [], []
    for i in range(len(value_seg_list_sort)):
        k += 1
        if k / len(value_seg_list_sort) <= inter_new: 
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
    plt.xlabel("Rank")
    plt.ylabel("Cumulative segregation values") # P(Seg$_{i}$>$seg$)
    plt.legend()
dd_all()

#SAVE
value_rank_cir_level_all_pd = pd.DataFrame(value_rank_cir_level_all)
value_rank_cir_level_all_pd.to_csv('./model_results/Lorenz_curve/'+str(region)+'_value_rank_cir_level_all.csv')
value_seg_cir_level_all_pd = pd.DataFrame(value_seg_cir_level_all)
value_seg_cir_level_all_pd.to_csv('./model_results/Lorenz_curve/'+str(region)+'_value_seg_cir_level_all.csv')
x_intersection_pd = pd.DataFrame(x_intersection)
x_intersection_pd.to_csv('./model_results/Lorenz_curve/'+str(region)+'_x_intersection.csv')

# %% draw 不同level的travel flow mapline 方法
def plotscale(ax, bounds, textcolor='black', fontsize=9, rect=[0.1, 0.1],
              accuracy='auto', unit="KM", compasssize=1):
    import math
    lon1 = bounds[0]
    lat1 = bounds[1]
    lon2 = bounds[2]
    lat2 = bounds[3]
    # 划定比例尺栅格所代表的距离
    if accuracy == 'auto':
        accuracy = (int((lon2 - lon1) / 0.0003 / 1000 + 0.5) * 1000)
    # 计算比例尺栅格的经纬度增加量大小Lon和Lat
    deltaLon = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))

    # 指北针、比例尺的位置
    a, c = rect
    b = 1 - a  # 0.88 1
    d = 1 - c
    # alon, alat = lon2 - 1, lat2 - 2.5  # wdh2
    alon, alat = -40, 52  # wdh2
    # 加比例尺
    from shapely.geometry import Polygon
    scale = gpd.GeoDataFrame(
        {'color': [(0, 0, 0), (1, 1, 1)],
         'geometry':
             [Polygon(
                 [(alon + deltaLon, alat), (alon + 5 * deltaLon, alat), (alon + 5 * deltaLon, alat + deltaLon * 0.4),
                  (alon + deltaLon, alat + deltaLon * 0.4)]),
                 Polygon([(alon + 5 * deltaLon, alat), (alon + 9 * deltaLon, alat),
                          (alon + 9 * deltaLon, alat + deltaLon * 0.4), (alon + 5 * deltaLon, alat + deltaLon * 0.4)])]
         })
    scale.plot(ax=ax, edgecolor=(0, 0, 0, 1), facecolor=scale['color'], lw=0.6)

    if (unit == 'KM') | (unit == 'km'):
        unit_length = 1000
    if (unit == 'M') | (unit == 'm'):
        unit_length = 1
    ax.text(alon + deltaLon, alat + deltaLon * 0.5, '0', color=textcolor, fontsize=fontsize, ha='center', va='bottom')
    ax.text(alon + 5 * deltaLon, alat + deltaLon * 0.5, str(int(4 * accuracy / unit_length)), color=textcolor,
            fontsize=fontsize, ha='center', va='bottom')
    ax.text(alon + 9 * deltaLon, alat + deltaLon * 0.5, str(int(8 * accuracy / unit_length)), color=textcolor,
            fontsize=fontsize, ha='center', va='bottom')
    ax.text(alon + 9.5 * deltaLon, alat + deltaLon * 0.5, unit, color=textcolor, fontsize=fontsize, ha='left', va='top')

    # 加指北针
    deltaLon = compasssize * deltaLon
    alon = alon - deltaLon
    compass = gpd.GeoDataFrame(
        {'color': [(0, 0, 0), (1, 1, 1)],
         'geometry':
             [Polygon([[alon, alat], [alon, alat + deltaLon], [alon + 1 / 2 * deltaLon, alat - 1 / 2 * deltaLon]]),
              Polygon([[alon, alat], [alon, alat + deltaLon], [alon - 1 / 2 * deltaLon, alat - 1 / 2 * deltaLon]])]
         })

def draw_method(odlc, flow, norm, cmap, tess_, level_num):
    bounds = tess_.total_bounds
    ss_ = 0.02
    bounds = np.array([bounds[0] - ss_, bounds[1] - ss_, bounds[2] + ss_, bounds[3] + ss_])
    fig, ax = plt.subplots(figsize=(12, 10))  # sharex=True, sharey=True,
    tess_.plot(ax=ax, facecolor='#D4EFEF', edgecolor='black', alpha=0.65)  # 画地图底图
    odlc.set_array(flow)  # 设置每条线的值，用于上色
    lw = 2 * (0.2 + 0.2 * norm(flow)) ** 0.5  # 设置线宽
    odlc.set_linewidth(lw)
    # 其它设置
    odlc.set_capstyle('round')
    # odlc.set_alpha(0.6)
    odlc.set_alpha(1)
    ax.add_collection(odlc)  # 画flow
    fig.colorbar(ax=ax, mappable=mpl.cm.ScalarMappable(cmap=cmap, norm=norm),  # 位置[左,下,右,上] #WDH
                 fraction=0.03, shrink=0.2, aspect=7, pad=-0.075, anchor=(-5, 0.3))  # 0.22-0.1-0.02   #0.4-0.3
    # 加指北针比例尺
    plotscale(ax, bounds, textcolor='black', fontsize=14, rect=[0.03, 0.85],  # rect:画在图中的位置 rect=[0.03,0.85]
              accuracy=10000, unit="KM", compasssize=3)  # 按accuracy的4倍画1格比例尺
    plt.rcParams['xtick.direction'] = 'in'  ####坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in'  ####坐标轴刻度朝内
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
    font_eng = {'family': 'Arial', 'weight': 'normal', 'size': 20}
    ax.set_xlabel('Longitude', font_eng)
    ax.set_ylabel('Latitude', font_eng)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlim(-150, -20)
    # 缩紧子图边距
    plt.savefig("./Figure/Flow_mapline_level_" + str(level_num) + ".png", dpi=360, bbox_inches='tight')

def draw2(od_flow, tess_, level_num):
    # 准备所有点的位置信息
    tile_case_lng_lat = []
    key_Set = {}
    for key, values in od_flow.items():
        if key in key_Set.keys():
            continue
        ind = []
        ind.append(tess_[tess_['GEOID'] == key].geometry.centroid.to_list()[0].x)
        ind.append(tess_[tess_['GEOID'] == key].geometry.centroid.to_list()[0].y)
        ind.append(str(key))
        tile_case_lng_lat.append(ind)
        key_Set[key] = 1
        for subkey, value in od_flow[key].items():
            if subkey in key_Set.keys():
                continue
            ind = []
            ind.append(tess_[tess_['GEOID'] == subkey].geometry.centroid.to_list()[0].x)
            ind.append(tess_[tess_['GEOID'] == subkey].geometry.centroid.to_list()[0].y)
            ind.append(str(subkey))
            tile_case_lng_lat.append(ind)
            key_Set[subkey] = 1

    tile_case_lng_lat_pd = pd.DataFrame(tile_case_lng_lat)
    tile_case_lng_lat_pd.columns = ['lon', 'lat', 'Location']
    print(max(tile_case_lng_lat_pd['lon']))

    # 可视化真实的flow的map distribution
    od_flow_real = []
    for key, values in od_flow.items():
        for subkey, value in od_flow[key].items():
            if key == subkey:
                continue
            ind = []
            ind.append(key)
            ind.append(subkey)
            ind.append(value)
            od_flow_real.append(ind)

    od_flow_real = pd.DataFrame(od_flow_real)
    od_flow_real.columns = ['origion', 'destination', 'Flow']

    od_flow_real = (
        od_flow_real
            .merge(tile_case_lng_lat_pd[['lon', 'lat', 'Location']], left_on='origion',
                   right_on='Location')  # o的位置拼到od_flow上
            .rename(columns={'lon': 'o_lng', 'lat': 'o_lat'})
            .drop(columns=['Location'])
            .merge(tile_case_lng_lat_pd[['lon', 'lat', 'Location']], left_on='destination',
                   right_on='Location')  # d的位置拼到od_flow上
            .rename(columns={'lon': 'd_lng', 'lat': 'd_lat'})
            .drop(columns=['Location'])
            .sort_values(['Flow', 'origion'])  # 按Flow从小到大排序，画图更好看
    )
    # 设置colormap
    flows = od_flow_real.Flow  # 取出real flow 和pred flow，以便找到最大值，从而归一化
    cmap = copy(mpl.cm.hot)  # hot
    norm = mpl.colors.LogNorm(vmin=1, vmax=flows.max())  # flows.max() #10000 1000 3000
    # 用LineCollection保存每个od，加快渲染速度
    od_pairs_real = np.array([[[od.o_lng, od.o_lat], [od.d_lng, od.d_lat]]
                              for _, od in od_flow_real.iterrows()])
    odr = LineCollection(od_pairs_real, cmap=cmap, norm=norm)
    # 画图数据准备
    odlc_list = [odr, odr]
    od_flow_list = [od_flow_real.Flow, od_flow_real.Flow]
    # 开始画图
    draw_method(odlc_list[0], od_flow_list[0], norm, cmap, tess_, level_num)
    # plt.show()

# 看相关性
o_flow_all, o_degree_all, loc_set = {}, {}, []
# 获得loc 集合
for key, values in o2d2flow_msa.items():
    loc_set.append(key)
    for skey, va in o2d2flow_msa[key].items():
        loc_set.append(skey)
loc_set = list(set(loc_set))

# 计算和存储各个loc的inflow和degree
for loc in loc_set:
    ss, dd_tot = 0, 0
    for key, values in o2d2flow_msa.items():
        for skey, va in o2d2flow_msa[key].items():
            if loc == skey:
                ss += va;
                dd_tot += 1
    o_flow_all[loc] = ss
    o_degree_all[loc] = dd_tot

# 为每个level 分配相应的od flow信息
o2d2flow_level_all = []
for i in range(len(loc_level_all)):
    ind = {}
    o2d2flow_level_all.append(ind)

for key, values in o2d2flow_msa.items():
    for i in range(len(loc_level_all)):
        if key in loc_level_all[i]:
            o2d2flow_level_all[i][key] = values
            break
        if i == len(loc_level_all)-1: #有些地点是没seg值的
            print(key)

# 分配相应的 flow信息（取log）、seg信息、以及degree给每个出发地O
o2d2flow_level_flow, o2d2flow_level_seg, o2d2flow_level_degree = [], [], []
region_all_info_set_dict = {}
for i in range(len(tess_selected_final)):
    g_id = tess_selected_final.loc[i]['GEOID']
    flag = 0
    for i in range(len(loc_level_all)):
        if g_id not in o_flow_all.keys(): #从一个区域出发无flow，也就是没这个区域的数据 因为遍历的是 tess_selected_final
            continue
        if o_flow_all[g_id] == 0:
            continue
        if g_id in loc_level_all[i]:
            o2d2flow_level_flow.append(np.log(o_flow_all[g_id]))
            o2d2flow_level_seg.append(location_segregation_value[g_id])
            o2d2flow_level_degree.append(o_degree_all[g_id])
            region_all_info_set_dict[g_id] = []
            region_all_info_set_dict[g_id].append(location_segregation_value[g_id]) # seg value
            region_all_info_set_dict[g_id].append(np.log(o_flow_all[g_id])) # o_flow
            region_all_info_set_dict[g_id].append(o_degree_all[g_id]) # degree
            flag = 1
            break
    if flag == 0:
        continue

# 计算相关性
from scipy import stats
pcc_flow_seg = stats.spearmanr(o2d2flow_level_flow, o2d2flow_level_seg)
pcc_degree_seg = stats.spearmanr(o2d2flow_level_degree, o2d2flow_level_seg)

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
    # data1 = np.array(data1).reshape(-1,1)
    # data2 = np.array(data2).reshape(-1,1)
    # ax.set_ylim(0, 30)
    if x_name=="degree":
        ax.set_xlim(0, 60)
    # ax.set_xlim(0, 100)
    # plt.scatter(data1,data2, cmap='jet', alpha=0.95, s=50, edgecolor='black')
    import seaborn as sns
    sns.kdeplot(data1, data2, cmap='Blues', fill=True)
    plt.xlabel(x_name, fontdict=font1)
    plt.ylabel(y_name, fontdict=font1)
    plt.xticks(fontsize=16)  # 横轴刻度字体大小
    plt.yticks(fontsize=16)  # 纵轴刻度字体大小
    plt.savefig("./Figure/" + str(fig_name) + ".png", dpi=360, bbox_inches='tight')

draw_scatter(o2d2flow_level_flow, o2d2flow_level_seg, "inflow(log)", "seg", 'log_inflow-seg')
draw_scatter(o2d2flow_level_degree, o2d2flow_level_seg, "degree", "seg", 'degree-seg')  # 60

o2d2flow_level_flow, o2d2flow_level_seg, o2d2flow_level_degree = [], [], []
for i in range(len(tess_selected_final)):
    g_id = tess_selected_final.loc[i]['GEOID']
    flag = 0
    for i in range(len(loc_level_all)):
        if g_id not in o_flow_all.keys(): #从一个区域出发无flow，也就是没这个区域的数据 因为遍历的是 tess_selected_final
            continue
        if o_flow_all[g_id] == 0:  # 从一个区域出发无flow，也就是没这个区域的数据 因为遍历的是 tess_selected_final
            continue
        if g_id in loc_level_all[i]:
            o2d2flow_level_flow.append(np.log(o_flow_all[g_id]))
            o2d2flow_level_seg.append(location_segregation_value[g_id])
            o2d2flow_level_degree.append(o_degree_all[g_id])
            flag = 1
            break
    if flag == 0:
        o2d2flow_level_flow.append(np.nan)
        o2d2flow_level_seg.append(np.nan)
        o2d2flow_level_degree.append(np.nan)

tess_selected_final['flow_level1'] = o2d2flow_level_degree
draw_utils.draw_corr(tess_selected_final, 'flow_level1', "degree", 50) #200/50
tess_selected_final['flow_level1'] = o2d2flow_level_flow
draw_utils.draw_corr(tess_selected_final, 'flow_level1', "flow_all", 6)
o2d2flow_level_seg_temp = [] # 此时，seg越大，表示隔离越小，讲道理flow会越大
for i in range(len(o2d2flow_level_seg)):
    o2d2flow_level_seg_temp.append(1 - o2d2flow_level_seg[i])  #***** 1表示不隔离,0表示隔离
tess_selected_final['flow_level1'] = o2d2flow_level_seg_temp
draw_utils.draw_corr(tess_selected_final, 'flow_level1', "seg", 1)

# draw2(o2d2flow_level1,tess_tract_all,1)
# draw2(o2d2flow_level2,tess_tract_all,2)
# draw2(o2d2flow_level3,tess_tract_all,3)
# draw2(o2d2flow_level4,tess_tract_all,4)
# draw2(o2d2flow_level5,tess_tract_all,5)
print('可视化相关性结束')

#%% 网络特征与个体出行特征
max_level_num = 5
loc_level_degree_all, loc_level_distance_all, loc_level_poiclass_all = [], [], []
for i in range(max_level_num):
    loc_level_degree_all.append([])
    loc_level_distance_all.append([])
    loc_level_poiclass_all.append({})

for key, va in o2d2flow_msa.items():
    if key not in location_segregation_value_all.keys():
        continue
    o_seg = location_segregation_value_all[key]
    o_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == key].iloc[0].geometry.centroid
    for i in range(max_level_num):
        if i==max_level_num-1:
            dis_ave = 0.
            for subkey, va in o2d2flow_msa[key].items():
                d_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == subkey].iloc[0].geometry.centroid
                dis = utils.earth_distance([o_lng_lat.x, o_lng_lat.y], [d_lng_lat.x, d_lng_lat.y])
                # dis = od_distance[(key,subkey)]
                loc_level_distance_all[max_level_num-1].append(dis)
                dis_ave += dis
            loc_level_degree_all[max_level_num-1].append(len(o2d2flow_msa[key]))
            if key in region_all_info_set_dict.keys():
                region_all_info_set_dict[key].append(5) #保存所属的层级
                region_all_info_set_dict[key].append(dis_ave*1.0/len(o2d2flow_msa[key])) #保存该region出发的平均dis
        else:
            if o_seg > x_intersection[i]:
                dis_ave = 0.
                for subkey, va in o2d2flow_msa[key].items():
                    d_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == subkey].iloc[0].geometry.centroid
                    dis = utils.earth_distance([o_lng_lat.x, o_lng_lat.y], [d_lng_lat.x, d_lng_lat.y])
                    # dis = od_distance[(key, subkey)]
                    loc_level_distance_all[i].append(dis)
                    dis_ave += dis
                loc_level_degree_all[i].append(len(o2d2flow_msa[key]))
                if key in region_all_info_set_dict.keys():
                    region_all_info_set_dict[key].append(i+1) #保存所属的层级
                    region_all_info_set_dict[key].append(dis_ave*1.0/len(o2d2flow_msa[key])) #保存该region出发的平均dis
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
    bin_loc_level_degree_ind,fre_loc_level_degree_ind = cal_his_figure(loc_level_degree_all[i], bbin)
    bin_loc_level_degree_all.append(bin_loc_level_degree_ind)
    fre_loc_level_degree_all.append(fre_loc_level_degree_ind)
draw_distribution_pre(bin_loc_level_degree_all,fre_loc_level_degree_all, "Traveling degree", region)

bbin = 10
bin_loc_level_distance_all,fre_loc_level_distance_all = [],[]
for i in range(len(loc_level_distance_all)):
    bin_loc_level_distance_ind,fre_loc_level_distance_ind = cal_his_figure(loc_level_distance_all[i], bbin)
    bin_loc_level_distance_all.append(bin_loc_level_distance_ind)
    fre_loc_level_distance_all.append(fre_loc_level_distance_ind)
draw_distribution_pre(bin_loc_level_distance_all,fre_loc_level_distance_all, "Traveling distance", region)

loc_level1_degree_pd = pd.DataFrame(loc_level_degree_all[0]);loc_level2_degree_pd = pd.DataFrame(loc_level_degree_all[1]);
loc_level3_degree_pd = pd.DataFrame(loc_level_degree_all[2]);loc_level4_degree_pd = pd.DataFrame(loc_level_degree_all[3]);
loc_level5_degree_pd = pd.DataFrame(loc_level_degree_all[4]);loc_level1_distance_pd = pd.DataFrame(loc_level_distance_all[0]);
loc_level2_distance_pd = pd.DataFrame(loc_level_distance_all[1]);loc_level3_distance_pd = pd.DataFrame(loc_level_distance_all[2]);
loc_level4_distance_pd = pd.DataFrame(loc_level_distance_all[3]);loc_level5_distance_pd = pd.DataFrame(loc_level_distance_all[4]);
loc_level1_degree_pd.to_csv("./model_results/Travel_features_data/Level1_degree_"+str(region)+".csv")
loc_level2_degree_pd.to_csv("./model_results/Travel_features_data/Level2_degree_"+str(region)+".csv")
loc_level3_degree_pd.to_csv("./model_results/Travel_features_data/Level3_degree_"+str(region)+".csv")
loc_level4_degree_pd.to_csv("./model_results/Travel_features_data/Level4_degree_"+str(region)+".csv")
loc_level5_degree_pd.to_csv("./model_results/Travel_features_data/Level5_degree_"+str(region)+".csv")
loc_level1_distance_pd.to_csv("./model_results/Travel_features_data/Level1_distance_"+str(region)+".csv")
loc_level2_distance_pd.to_csv("./model_results/Travel_features_data/Level2_distance_"+str(region)+".csv")
loc_level3_distance_pd.to_csv("./model_results/Travel_features_data/Level3_distance_"+str(region)+".csv")
loc_level4_distance_pd.to_csv("./model_results/Travel_features_data/Level4_distance_"+str(region)+".csv")
loc_level5_distance_pd.to_csv("./model_results/Travel_features_data/Level5_distance_"+str(region)+".csv")

bin_loc_level_degree_all_pd = pd.DataFrame(bin_loc_level_degree_all)
fre_loc_level_degree_all_pd = pd.DataFrame(fre_loc_level_degree_all)
bin_loc_level_degree_all_pd.to_csv("./model_results/Travel_features_value/bin_level_degree_all_"+str(region)+".csv")
fre_loc_level_degree_all_pd.to_csv("./model_results/Travel_features_value/fre_level_degree_all_"+str(region)+".csv")
bin_loc_level_distance_all_pd = pd.DataFrame(bin_loc_level_distance_all)
fre_loc_level_distance_all_pd = pd.DataFrame(fre_loc_level_distance_all)
bin_loc_level_distance_all_pd.to_csv("./model_results/Travel_features_value/bin_level_distance_all_"+str(region)+".csv")
fre_loc_level_distance_all_pd.to_csv("./model_results/Travel_features_value/fre_level_distance_all_"+str(region)+".csv")

#%% 统计不同O出发的poi访问次数
o_dpoi_flow = {}  # 140414
poi_class_set = []
poi_class_set_dict = {}
for key, value in user_in_MSA_final.items(): #统计poi类别信息
    for i in range(1, len(value)):
        o_county_id = str(value[i - 1][4]) + str(value[i - 1][5]) + str(value[i - 1][7])  # +str(value[i-1][8])
        if o_county_id not in o_dpoi_flow.keys():
            o_dpoi_flow[o_county_id] = {}
        d_county_id = str(value[i][4]) + str(value[i][5]) + str(value[i][7])  # +str(value[i][8])
        d_poi_class = str(value[i][2])
        if d_county_id not in poi_class_set_dict.keys():
            poi_class_set_dict[d_county_id] = []
        else:
            poi_class_set_dict[d_county_id].append(d_poi_class)
        d_poi_class = d_poi_class[:d_poi_class.find(":")+1]
        d_poi_class = d_poi_class.replace(":","")
        if d_poi_class== "" :
            continue
        if d_poi_class=='Home / Work / Other':
            d_poi_class = 'Homes, Work, Others'
        if d_poi_class=='Nightlife Spots':
            d_poi_class = 'Nightlife'
        if d_poi_class=='Colleges & Universities':
            d_poi_class = 'College & Education'
        if d_poi_class=='Travel Spots':
            d_poi_class = 'Travel'
        poi_class_set.append(d_poi_class)
        if d_poi_class not in o_dpoi_flow[o_county_id].keys():
            o_dpoi_flow[o_county_id][d_poi_class] = 1
        else:
            o_dpoi_flow[o_county_id][d_poi_class] += 1

for key,values in poi_class_set_dict.items():
    poi_class_set_dict[key] = list(set(values))
# 
# poi_class_set = list(set(poi_class_set))
# poi_class_set_pd = pd.DataFrame(poi_class_set)
# poi_class_set_pd.to_csv("./model_results/poi_class_set_"+str(region)+".csv")
pd_ref = pd.read_csv("./model_results/poi_class_set_NY.csv",header=0,index_col=0).values.reshape(-1)

poi_class_set_dict = {}
for i in range(len(pd_ref)):
    poi_class_set_dict[pd_ref[i]] = i

for key, values in o_dpoi_flow.items():
    if key in location_segregation_value_all.keys():
        o_seg = location_segregation_value_all[key]
        for i in range(max_level_num):
            if i == (max_level_num-1):
                loc_level_poiclass_all[max_level_num-1][key] = values
            else:
                if o_seg > x_intersection[i]:
                    loc_level_poiclass_all[i][key] = values
                    break

#统计部分level下poi访问的平均分布
loc_level_poi_num_all = []
for i in range(max_level_num):
    loc_level_poi_num_all.append(np.zeros(9))

for level_num in range(max_level_num):
    for key, values in loc_level_poiclass_all[level_num].items(): #values 存储了不同key出行的人，的所有poi访问。O是定的。
        ss = 0
        for subkey,va in values.items():
            ss += va
        for subkey,va in values.items(): #从每个位置出发的poi访问类别、和访问次数
            loc_level_poi_num_all[level_num][poi_class_set_dict[subkey]] += va/ss
for i in range(max_level_num):
    if len(loc_level_poiclass_all[i])!=0:
        loc_level_poi_num_all[i] = loc_level_poi_num_all[i]/len(loc_level_poiclass_all[i]) #平均到人

#拼接
loc_level_all_poinum = loc_level_poi_num_all[0].reshape(1, -1)
for i in range(1, max_level_num):
    loc_level_all_poinum = np.vstack((loc_level_all_poinum, loc_level_poi_num_all[i].reshape(1, -1)))
loc_level_all_poinum_sum = np.max(loc_level_all_poinum, axis=0)
for i in range(len(loc_level_all_poinum)):
    for j in range(len(loc_level_all_poinum[i])):
        loc_level_all_poinum[i][j] /= loc_level_all_poinum_sum[j]

# 可视化各个block 4个level的比例
def draw_heatmap_poiclass(data, name, cc):
    font1 = {'family': 'Arial', 'color': 'Black', 'size': 18}
    f, ax = plt.subplots(figsize=(9, 8))
    plt.imshow(data, cmap=cc, aspect=1)
    cb = plt.colorbar(shrink=0.5)
    cb.ax.tick_params(labelsize=17)  # 设置色标刻度字体大小。
    sh = plt.gca()
    y = range(1, 5)
    x = range(0, 9)
    plt.xticks(x,poi_class_set,rotation=90)
    plt.yticks(y)
    plt.xticks(fontsize=16)  # 横轴刻度字体大小
    plt.yticks(fontsize=17)  # 纵轴刻度字体大小
    sh.set_ylabel("Seg levels", fontdict=font1)  # 横轴名称
    sh.set_xlabel("Poi categories", fontdict=font1)  # 纵轴名称
    plt.savefig("./Figure/Poi_visiting_" + str(name) + ".png", dpi=360, bbox_inches='tight')
    plt.show()

poi_class_set = pd_ref
draw_heatmap_poiclass(loc_level_all_poinum,region,'Reds')

# %% 分析5个level之间跨层次访问模式，热图矩阵呈现
seg_to_seg_adj = np.zeros((5, 5))
seg_to_seg_dis_adj = np.zeros((5, 5))
kk=0
for (o_id, d_id), va in odflow_msa.items(): #计算2个seg_level之间的dis、flow，然后写入adj矩阵中
    if (o_id not in location_segregation_value_all.keys()) or (d_id not in location_segregation_value_all.keys()):
        kk+=1
        continue
    o_seg, d_seg = location_segregation_value_all[o_id], location_segregation_value_all[d_id]
    o_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == o_id].iloc[0].geometry.centroid
    d_lng_lat = tess_tract_selected[tess_tract_selected['GEOID'] == d_id].iloc[0].geometry.centroid
    dis = utils.earth_distance([o_lng_lat.x, o_lng_lat.y], [d_lng_lat.x, d_lng_lat.y])
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
seg_to_seg_adj_pd.to_csv("./model_results/CLVI_results/CLVI_adj_"+str(region)+".csv")
seg_to_seg_dis_adj_pd.to_csv("./model_results/CLVI_results/CLVI_dis_adj_"+str(region)+".csv")

for i in range(5):
    for j in range(5):
        if seg_to_seg_adj_save[i][j] != 0:
            seg_to_seg_dis_adj_save[i][j] = seg_to_seg_dis_adj_save[i][j] / seg_to_seg_adj_save[i][j]

seg_to_seg_adj_save_nor = seg_to_seg_adj_save.copy()
for i in range(5):
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
    sh.set_xlabel("Level_D", fontdict=font1)  # 横轴名称
    sh.set_ylabel("Level_O", fontdict=font1)  # 纵轴名称
    plt.savefig("./Figure/Seg_to_seg_adj_" + str(name) + ".png", dpi=360, bbox_inches='tight')
    # plt.show()

draw_heatmap(seg_to_seg_adj_save_nor, "seg_adj_nor", 'Reds')
draw_heatmap(seg_to_seg_dis_adj_save_nor, "dis_adj_nor", 'Reds')
seg_to_seg_adj_save_nor_pd = pd.DataFrame(seg_to_seg_adj_save_nor)
seg_to_seg_adj_save_nor_pd.to_csv("./model_results/CLVI_results/CLVI_adj_nor_"+str(region)+".csv")
seg_to_seg_dis_adj_save_nor_pd = pd.DataFrame(seg_to_seg_dis_adj_save_nor)
seg_to_seg_dis_adj_save_nor_pd.to_csv("./model_results/CLVI_results/CLVI_dis_adj_nor_"+str(region)+".csv")

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

# %% 分析5个level之间的Shannon entropy
max_level_num = 5
out_flow_level_all,out_flow_all_key, out_flow_all = [],[],[]
se_list_all_dict = {}
se_list_level_all, se_list_all = [], []
for i in range(max_level_num):
    out_flow_level_all.append([])
    se_list_level_all.append([])

for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():
        seg_v = location_segregation_value[key]
        sum_ = 0
        for subkey, va in o2d2flow_msa[key].items():
            sum_ += va
        ind = []
        for subkey, va in o2d2flow_msa[key].items():
            ind.append(va / sum_)
        out_flow_all.append(ind)
        out_flow_all_key.append(key)
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
    se_list_all_dict[out_flow_all_key[i]] = se  # 保存该loc出发的出行熵

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
    ind.to_csv('./model_results/Travel_features_value/Travel_entropy_Level'+str(i)+"_"+str(region)+".csv")
se_list_all_fre_sort_df.to_csv('./model_results/Travel_features_value/Travel_entropy_All'+"_"+str(region)+".csv")

# se_list_fre_sort_df_allx = []
# for i in range(max_level_num):
#     ind = pd.read_csv('./model_results/Travel_features_value/Travel_entropy_Level'+str(i)+"_"+str(region)+".csv")
#     se_list_fre_sort_df_allx.append(ind)
# se_list_all_fre_sort_dfx = pd.read_csv('./model_results/Travel_features_value/Travel_entropy_All'+"_"+str(region)+".csv")

plot = plt.figure()
ax1 = plot.add_subplot(1, 1, 1)
for i in range(max_level_num):
    ax1.plot(se_list_fre_sort_df_all[i]['Rds'], se_list_fre_sort_df_all[i]['cumsum'], label="Level "+str(i))
ax1.plot(se_list_all_fre_sort_df['Rds'], se_list_all_fre_sort_df['cumsum'], color='black', label="All")
plt.xlabel("Travel entropy $E$")
plt.ylabel("$P$(E$_{i}$>$E$)")
plt.legend()
# plt.savefig("./Figure/Entropy_of_different_Seg_levels_" + str(region) + ".png", dpi=360, bbox_inches='tight')
plt.show()

# %% 对出行数据进行聚类系数分析
out_flow_level_all, out_flow_all = [], []
c_list_level_all, c_all, c_all_dict = [], [], {}

for i in range(max_level_num):
    out_flow_level_all.append([])
    c_list_level_all.append([])

max_,min_ = -1,99999999
for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():
        for subkey, va in o2d2flow_msa[key].items():
            max_ = max(max_, va)
            min_ = min(min_, va)

od_flow_new = {}
for key, values in o2d2flow_msa.items():
    if key in location_segregation_value.keys():
        od_flow_new[key] = {}
        for subkey, va in o2d2flow_msa[key].items():
            od_flow_new[key][subkey] = va

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
        c_all_dict[key] = ci
        for i in range(max_level_num):
            if seg_v > x_intersection[i]:
                c_list_level_all[i].append(ci)
                break
            if i==max_level_num-1:
                c_list_level_all[i].append(ci)
        min_ = min(min_, ci)
        max_ = max(max_, ci)
    else:
        c_all_dict[key] = 1

c_list1_fre_sort_df_all =[]
for i in range(max_level_num):
    c_list1_fre_sort_df_all.append(get_cdf(c_list_level_all[i]))
c_all_fre_sort_df = get_cdf(c_all)

for i in range(max_level_num):
    ind = c_list1_fre_sort_df_all[i]
    ind.to_csv('./model_results/Travel_features_value/Clustering_coefficient_Level'+str(i)+"_"+str(region)+".csv")
c_all_fre_sort_df.to_csv('./model_results/Travel_features_value/Clustering_coefficient_All'+"_"+str(region)+".csv")

# 在开头插入新行数据
# c_list1_fre_sort_df = pd.concat([pd.DataFrame([new_row]), c_list1_fre_sort_df], ignore_index=True)
plot = plt.figure()
ax1 = plot.add_subplot(1, 1, 1)
for i in range(max_level_num):
    ax1.plot(c_list1_fre_sort_df_all[i]['Rds'], c_list1_fre_sort_df_all[i]['cumsum'], label="Level "+str(i))
ax1.plot(c_all_fre_sort_df['Rds'], c_all_fre_sort_df['cumsum'], color='black', label="All")
plt.xlabel("Clustering coefficient $C$")
plt.ylabel("$P$(C$_{i}$>$C$)")
ax1.set_xlim(-0.1, 8)
plt.legend()
# plt.savefig("./Figure/Clustering_coefficient_of_different_Seg_levels_" + str(region) + ".png", dpi=360,
#             bbox_inches='tight')
plt.show()

#%% 加入pop、income、land、road信息
for key,values in region_all_info_set_dict.items():
    if key in poi_class_set_dict.keys(): #POI
        region_all_info_set_dict[key].append(len(poi_class_set_dict[key]))
    else:
        region_all_info_set_dict[key].append(0)
    region_all_info_set_dict[key].append(se_list_all_dict[key]) #出行熵
    region_all_info_set_dict[key].append(c_all_dict[key]) #聚类系数
    region_all_info_set_dict[key].append(tess_tract_selected[tess_tract_selected["GEOID"] == key].POPULATION.values[0]) #人口
    region_all_info_set_dict[key].append(tess_tract_selected[tess_tract_selected["GEOID"] == key].INCOME.values[0]) #收入
    #land
    #...

with open('./dataset/region_results/region_all_info_set_dict_'+str(region)+'.json', 'w') as f:
    json.dump(region_all_info_set_dict, f) #

#%%

from scipy import stats
d_clvi = [-5.78,-5.89,-5.68,-6.53,-5.60,-4.54,-3.79,-3.19,-3.73,-5.42]
d_pop = [7.35,7.27,7.00,6.96,6.92,6.87,6.84,6.84,6.69,6.61]#pop
d_income = [5.08,4.92,5.04,5.07,5.35,4.96,4.89,4.92,5.07,5.00]#income
d_land = [0.656,0.648,1.142,0.930,1.072,0.925,1.133,0.585,1.242,1.042]#land
d_road = [12.672,11.300,9.254,11.424,9.438,10.177,6.660,11.123,8.320,8.773]#road
d_area = [4.62720672,5.005527648,4.548656149,4.592567034,4.643349363,4.300110566,4.51745785,4.490020786,4.46008223,4.47611943]
spcc_pop = stats.spearmanr(d_clvi, d_pop)
pcc_pop = stats.pearsonr(d_clvi, d_pop)
spcc_income = stats.spearmanr(d_clvi, d_income)
pcc_income = stats.pearsonr(d_clvi, d_income)
spcc_land = stats.spearmanr(d_clvi, d_land)
pcc_land = stats.pearsonr(d_clvi, d_land)
spcc_road = stats.spearmanr(d_clvi, d_road)
pcc_road = stats.pearsonr(d_clvi, d_road)
spcc_area = stats.spearmanr(d_clvi, d_area)
pcc_area = stats.pearsonr(d_clvi, d_area)

