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

#%% 加载数据 all data
dataset_all = pd.read_csv("./dataset/weeplace_checkins.csv",header=0) #7658368
dataset_all.columns=['uid','placeid','time','lat','lng','city','poi']

#%% 地理区划
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
# tess_tract_all=tess_tract_all.reset_index(drop=True)
# tess_tract_all.to_csv('./dataset/mapdata/tess_tract_all.csv')
tess_tract_all = pd.read_csv('./dataset/mapdata/tess_tract_all.csv',index_col=0)
tess_tract_all = gpd.GeoDataFrame(tess_tract_all)
from shapely import wkt
tess_tract_all['geometry']= tess_tract_all['geometry'].apply(wkt.loads)
end_time = datetime.datetime.now()

#%% seg与不同变量之间的相关性
with open('./dataset/tract_info_all_combine2.json', 'r', encoding='UTF-8') as f:
    tract_info_all_combine = json.load(fp=f)
#seg value\o_flow_log\degree\level_num\ave_dis\poi_class_num\Entropy\Cluster_eff\pop\income\road_density\land_use

level_all_x = [{},{},{},{},{}]
level_all_seg = [{},{},{},{},{}]
for key,values in tract_info_all_combine.items():
    level_all_seg[values[3]-1][key] = values[0]
    variable_ = values.copy()
    variable_.pop(0);  variable_.pop(1);
    level_all_x[values[3] - 1][key] = variable_
#o_flow_log\degree\ave_dis\poi_class_num\Entropy\Cluster_eff\pop\income\road_density\land_use

sample_x,sample_y = [],[]
for key,values in level_all_x[2].items():
    sample_x.append(values)
for key,values in level_all_seg[2].items():
    sample_y.append(values)

sample_x = np.array(sample_x)
sample_y = np.array(sample_y)
sample_x_max = np.max(sample_x, axis = 0)

for i in range(len(sample_x)):
    for j in range(10):
        sample_x[i][j] /= sample_x_max[j]

#构建分类模型
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
sample_x = sample_x.reshape(-1, 10)
sample_y = sample_y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(sample_x, sample_y, test_size=0.4, random_state=0)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor,AdaBoostRegressor
model = RandomForestRegressor(max_depth=5,n_estimators=1000) #
model.fit(X_train,y_train)

# 可视化分析
import shap
import matplotlib.pyplot as plt
import pandas as pd
feature_name = pd.read_csv("./dataset/Feature_name.csv").values[:, 1].tolist()
shap.initjs()  # 用来显示的
explainer = shap.Explainer(model)
shap_values = explainer(X_test, check_additivity=False)

plt.figure(dpi=360)
fig = plt.gcf()
shap.summary_plot(shap_values=shap_values,
                  features=X_test,
                  feature_names=feature_name,
                  plot_type='dot',
                  show=False)
fig.set_facecolor('white')  # 设置背景为白色
shap_values.base_values
plt.show()


with open('./dataset/tract_info_all_combine2.json', 'r', encoding='UTF-8') as f:
    tract_info_all_combine = json.load(fp=f)
#seg value\o_flow_log\degree\level_num\ave_dis\poi_class_num\Entropy\Cluster_eff\pop\income\road_density\land_use

level_all_x = []
level_all_level = []
for key,values in tract_info_all_combine.items():
    level_all_level.append(values[3]-1)
    variable_ = values.copy()
    variable_.pop(0);variable_.pop(0);variable_.pop(0); variable_.pop(0);
    level_all_x.append(variable_)

sample_x = np.array(level_all_x)
sample_y = np.array(level_all_level)
sample_x_max = np.max(sample_x, axis = 0)
sample_x_min = np.min(sample_x, axis = 0)

for i in range(len(sample_x)):
    for j in range(len(sample_x[i])):
        # sample_x[i][j] /= sample_x_max[j]
        if sample_x[i][j]<=0:
            sample_x[i][j]=0

# 构建分类模型
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingRegressor,AdaBoostRegressor,GradientBoostingRegressor,StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

sample_x = sample_x.reshape(-1, 8)
sample_y = sample_y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(sample_x, sample_y, test_size=0.2, random_state=0)

model = RandomForestClassifier(max_depth=5,n_estimators=100) #
model.fit(X_train,y_train)

#%%
import shap
import matplotlib.pyplot as plt
import pandas as pd
feature_name = pd.read_csv("./dataset/Feature_name.csv").values[2:, 1].tolist()

shap.initjs()  # 用来显示的
explainer = shap.Explainer(model)
# 对特征重要度进行解释
shap_values_ = explainer.shap_values(X_test)
plt.figure(dpi=360)
fig = plt.gcf()

import matplotlib.cm as cm
# bar_colors = ['red', 'green', 'blue', 'orange', 'yellow']  # 替换为你想要的颜色
color_map = cm.get_cmap("bwr")  # 替换为你喜欢的颜色映射
color_func = lambda i: color_map(i / len(shap_values_))

shap.summary_plot(shap_values=shap_values_,
                 features=X_test,
                 feature_names=feature_name,
                 plot_type='bar',
                 color=color_func)
fig.set_facecolor('white')  # 设置背景为白色
plt.show()

# 计算柱的均值
bar_heights = shap_values_[4].mean(0)
# 绘制 Shap 柱状图
plt.bar(range(len(bar_heights)), bar_heights)
# 显示图表
plt.show()

#%%