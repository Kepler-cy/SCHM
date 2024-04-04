import math
import json
import pandas as pd
import numpy as np
import json
import datetime
# from skmob.tessellation import tilers
import geopandas as gpd
import matplotlib.pyplot as plt
from math import sqrt, sin, cos, pi, asin
import matplotlib as mpl

font1 = {'family':'Arial','color': 'Black','size':16}
def draw_corr(tess,col_name,savename,maxxx):
    #设置图片的大小
    figsize = 9,9
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
    sh.set_xlabel("Longitude",fontdict=font1) #横轴名称
    sh.set_ylabel("Latitude",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    tess.plot(ax=ax, column=col_name, cmap='bwr',vmin=0,vmax=maxxx) #, cmap='jet',edgecolor='gray',       , column='ar'  ,vmin=1,vmax=5
    plt.savefig("./Figure/Flow_level_all_"+str(savename)+"_.png", dpi=360, bbox_inches="tight")
    plt.show()

def draw1(tess,col_name):
    font1 = {'family':'Arial','color': 'Black','size':16}
    #设置图片的大小
    figsize = 9,9
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
    sh.set_xlabel("Longitude",fontdict=font1) #横轴名称
    sh.set_ylabel("Latitude",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    tess.plot(ax=ax, edgecolor='gray',column=col_name, cmap='jet') #, cmap='jet'        , column='area'
    plt.show()

#%%
def draw2(tess,min_, max_,name,dataset_name,col_name):
    color_ = "Oranges"  #GnBu jet Greens  terrain  YlGn  Oranges
    font1 = {'family':'Arial','color': 'Black','size':16}
    # font3 = {'family':'Arial','color': 'Black','size':13}
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
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    # sh=plt.gca()
    # sh.set_xlabel("Longitude",fontdict=font1) #横轴名称
    # sh.set_ylabel("Latitude",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=0) #横轴刻度字体大小
    plt.yticks(fontsize=0) #纵轴刻度字体大小
    # ploy1= ploy.values.bounds #属于左下角和右上角2个坐标
    tess.plot(ax=ax, edgecolor='gray',column=col_name, cmap=color_) #, cmap='jet'
    fig = figure
    if name=="wdh":
        cax = fig.add_axes([0.8, 0.27, 0.032, 0.15])
    if col_name=='seg_value':
        sm = mpl.cm.ScalarMappable(cmap=color_)
    else:
        norm = mpl.colors.LogNorm(vmin=min_, vmax = max_)
        sm = mpl.cm.ScalarMappable(cmap=color_, norm=norm)
    sm._A = []
    bar = fig.colorbar(sm, cax=cax)
    bar.ax.tick_params(labelsize=14)  #设置色标刻度字体大小。
    plt.savefig("./Figure/USA_Oi_flow_"+str(dataset_name)+"_"+str(name)+".png", dpi=360, bbox_inches = 'tight')

#%%
font1 = {'family' : 'Arial','color':'Black','size': 16}
def draw_distribution_data(datax,datay,dis_name,region):
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
    ax.set_yscale("log")
    if dis_name=="Traveling distance":
        sh.set_xlabel("$\it{d}$ (km)",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{d}$)",fontdict=font1) #纵轴名称
        # if region=="wdh":
        #     ax.set_xlim(10, 2000)
        # else:
        #     ax.set_xlim(10, 1000)
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
        if region=="wdh":
            ax.set_xlim(10, 2000)
        else:
            ax.set_xlim(10, 1000)
        if region=="uk":
            ax.set_xlim(9, 1000)
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    # plt.scatter(datax,datay,linewidth=1.2,c='black',label="Real")
    plt.plot(datax,datay,linewidth=1.2,c='green',label="Real",marker='o',markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    plt.subplots_adjust(left=0,right=0.9)
    # plt.legend(loc="upper right")
    plt.savefig("./Figure/Data_distribution "+str(dis_name)+"_"+str(region)+".png",dpi=360,bbox_inches = 'tight')
    
#%%
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
    ax.set_yscale("log")
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
    # plt.scatter(datax,datay,linewidth=1.2,c='black',label="Real")
    plt.plot(datax,datay,linewidth=1.2,c='green',label="Real",marker='o',markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    plt.subplots_adjust(left=0,right=0.9)
    # plt.legend(loc="upper right")
    plt.savefig("./Figure/Data_distribution "+str(dis_name)+"_"+str(region)+".png",dpi=360,bbox_inches = 'tight')

#%%
font1 = {'family' : 'Arial','color':'Black','size': 16}
def draw_distribution(datax,datay,datax1,datay1,dis_name,region):
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
    # if dis_name!="Traveling steps":
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    if dis_name=="Traveling distance":
        sh.set_xlabel("$\it{d}$ (km)",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{d}$)",fontdict=font1) #纵轴名称
        # if region=='wdh':
        #     ax.set_xlim(10, 3000)
        # else:
        #     ax.set_xlim(10, 1000)
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
        # sh.set_xlabel("Traveling radius",fontdict=font1) #横轴名称
        sh.set_xlabel("$\it{r}$ (km)",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{r}$)",fontdict=font1) #纵轴名称
        mm_arker = '^'
        # ax.set_xlim(0, 1000)
    elif dis_name=="Traveling steps":
        # sh.set_xlabel("Traveling radius",fontdict=font1) #横轴名称
        sh.set_xlabel("$\it{T}$",fontdict=font1) #横轴名称
        sh.set_ylabel("$P(\it{T}$)",fontdict=font1) #纵轴名称
        mm_arker = '^'
        # ax.set_xlim(10, 1000)
        # if region=='uk':
        #     ax.set_xlim(5, 1000)
        
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    # plt.scatter(datax,datay,linewidth=1.2,c='orange',label="Real",marker='s')
    # plt.scatter(datax1,datay1,linewidth=1.2,c='green', label="Simulation",marker='o')
    plt.plot(datax,datay,linewidth=1.2,c='black',label="Real",marker='o',markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    plt.plot(datax1,datay1,linewidth=1.2,c='green', label="Simulation",marker=mm_arker,markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    plt.subplots_adjust(left=0,right=0.9)
    plt.legend(loc="upper right")
    plt.savefig("./Figure/Data_distribution "+str(dis_name)+"_"+str(region)+".png",dpi=360,bbox_inches = 'tight')
    
    
#%%
def draw_loss(d,g,title):
    font1 = {'family':'Arial','color': 'Black','size':16}
    font3 = {'family':'Arial', 'size':16}
    #设置图片的大小
    figsize = 9,9
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
    sh.set_xlabel("Epoch",fontdict=font1) #横轴名称
    sh.set_ylabel("Loss",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    idd = np.arange(1,len(d)+1,1)
    plt.plot(idd,d,linewidth=1.2,c='b',label="D_loss")
    plt.plot(idd,g,linewidth=1.2,c='g',label="G_loss")
    plt.title(title)
    plt.legend(loc="upper right",prop=font3,edgecolor='black')
    plt.savefig("./Loss.png",dpi=360,bbox_inches = 'tight')
#%%
def draw_loss_mscnn(d,g,title):
    font1 = {'family':'Arial','color': 'Black','size':16}
    font3 = {'family':'Arial', 'size':16}
    #设置图片的大小
    figsize = 9,9
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
    sh.set_xlabel("Epoch",fontdict=font1) #横轴名称
    sh.set_ylabel("Loss",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    idd = np.arange(1,len(d)+1,1)
    plt.plot(idd,d,linewidth=1.2,c='b',label="County_loss")
    plt.plot(idd,g,linewidth=1.2,c='g',label="Community_loss")
    plt.title(title)
    plt.legend(loc="upper right",prop=font3,edgecolor='black')
    plt.savefig("./Loss.png",dpi=360,bbox_inches = 'tight')
    
#%%
font1 = {'family' : 'Arial','color':'Black','size': 16}
def draw_distribution2(datay,datay1,datax,datax1,dis_name):
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
    ax.set_yscale("log")
    sh.set_xlabel("$\it{t}$",fontdict=font1) #横轴名称
    sh.set_ylabel("$S(\it{t}$)",fontdict=font1) #纵轴名称
    mm_arker = 'D'
    ax.set_xlim(0,len(datax1))
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    plt.scatter(datax,datay,linewidth=1.2,c='Orange',label="Real",marker='o')
    plt.scatter(datax1,datay1,linewidth=1.2,c='green', label="Simulation",marker='D')
    # plt.plot(datax,datay,linewidth=1.2,c='black',label="Real",marker='o',markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    # plt.plot(datax1,datay1,linewidth=1.2,c='green', label="Simulation",marker=mm_arker,markersize=8,markeredgecolor='black',markeredgewidth=0.5)
    plt.subplots_adjust(left=0,right=0.9)
    plt.legend(loc="upper right")
    plt.savefig("./Figure/Distribution visiting time.png",dpi=360,bbox_inches = 'tight')
#%%
def cal_his_figure(dataset,numBins,bin_edges):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fre_real_same, bins_real_same, patches  = ax.hist(dataset, bins=bin_edges,color='blue',alpha=0.8,rwidth=0.9,align = 'mid', histtype='bar')
    fre_real_dif, bins_real_dif, patches  = ax.hist(dataset, bins = numBins,color='blue',alpha=0.8,rwidth=0.9,align = 'mid', histtype='bar')
    
    bin_real_same = np.zeros(numBins)
    for i in range(len(bin_edges)-1):
        bin_real_same[i]=(bin_edges[i]+bin_edges[i+1])/2.0
    sum_=np.sum(fre_real_same)
    for i in range(len(fre_real_same)):
        fre_real_same[i] = fre_real_same[i]/sum_
        
    bin_real_dif = np.zeros(numBins)
    for i in range(len(bins_real_dif)-1):
        bin_real_dif[i]=(bins_real_dif[i]+bins_real_dif[i+1])/2.0
    sum_=np.sum(fre_real_dif)
    for i in range(len(fre_real_dif)):
        fre_real_dif[i] = fre_real_dif[i]/sum_
        
    return bin_real_dif,fre_real_dif,bin_real_same,fre_real_same

#%%
def draw_density(metric_id,metric_name,data,region):
    import seaborn as sns 
    font2 = {'family' : 'Arial','size': 13}
    sns.set()  #切换到sns的默认运行配置
    sns.set_style(style='white')
    #设置图片的大小
    figsize = 8,5
    plt.rcParams['xtick.direction'] = 'in' ####坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in' ####坐标轴刻度朝内
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='x',length =4,width=1, which='major',bottom='on') #,top=True
    ax.tick_params(axis='x',length =4, width=1, which='major',top='on')#,right=True
    ax.tick_params(axis='y',length =8,width=1, which='major', left='on') #,top=True
    ax.tick_params(axis='y',length =4,width=1, which='major', right='on') #,right=True
    ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.ylabel("Frequency",fontdict=font1) #横轴名称
    plt.xlabel(str(metric_name),fontdict=font1) #纵轴名称
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    # ['royalblue','orange','#aadb3d']
    # cpc_gm_ind = sns.kdeplot(data[:,metric_id],  label = "GM",   bw_method=0.5, ax=ax, color = "royalblue",linewidth=1.2,shade=False)
    # cpc_rm_ind = sns.kdeplot(data[:,metric_id+3], label = "RM", bw_method=0.5, ax=ax, color = "green",  linewidth=1.2,shade=False)
    # cpc_ggan_ind = sns.kdeplot(data[:,metric_id+6], label = "GGAN",bw_method=0.5, ax=ax, color = "red",  linewidth=1.2,shade=False)
        
    plt.hist(data[:,metric_id], density=False, bins=60,label = "GM",color = "royalblue")
    plt.hist(data[:,metric_id+3], density=False, bins=60,label = "RM",color = "green",alpha=1)
    plt.hist(data[:,metric_id+6], density=False, bins=60,label = "PWO", color = "orange",alpha=1)
    plt.hist(data[:,metric_id+9], density=False, bins=60,label = "GGAN", color = "red",alpha=0.75)
    # plt.rcParams.update({'font.size':15})
    
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.99, 0.99),prop=font2)
    # cpc_gm_ind = sns.distplot(data[:,metric_id],  label = "GM", ax=ax, color = "royalblue")
    # cpc_rm_ind = sns.distplot(data[:,metric_id+3], label = "RM", ax=ax, color = "green")
    # cpc_ggan_ind = sns.distplot(data[:,metric_id+6], label = "GGAN", ax=ax, color = "red")
    # plt.legend(loc="upper right")
    if region=='bsh':
        ax.set_ylim(0,12)
    if region=='wdh':
        ax.set_ylim(0,10)
    if region=='uk':
        ax.set_ylim(0,10)
    
    # plt.show()
    # import copy
    # gm= copy.deepcopy(list(data[:,metric_id]))
    # #gm.sort();gm.reverse()
    # # print(len(gm))
    # plt.plot(np.arange(1,len(gm)+1),gm,linewidth=1.2,c='royalblue',label="GM")
    # rm = copy.deepcopy(list(data[:,metric_id+3]))
    # #rm.sort();rm.reverse()
    # plt.plot(np.arange(1,len(gm)+1),rm,linewidth=1.2,c='green',label="RM")
    # ggan= copy.deepcopy(list(data[:,metric_id+6]))
    # #ggan.sort();ggan.reverse()
    # plt.plot(np.arange(1,len(gm)+1),ggan,linewidth=1.2,c='red',label="GGAN")
    # plt.subplots_adjust(left=0,right=0.9)
    # plt.legend(loc="upper right")
    
    # if metric_name=="MAE":
    #     # 绘制子图
    #     axins = ax.inset_axes((0.38, 0.25, 0.45, 0.45))
    #     cpc_gm_ind = sns.kdeplot(data[:,metric_id],  bw_method=0.5, ax=axins, color = "royalblue",linewidth=1.2,shade=False)
    #     cpc_rm_ind = sns.kdeplot(data[:,metric_id+3], bw_method=0.5, ax=axins, color = "green",  linewidth=1.2,shade=False)
    #     cpc_ggan_ind = sns.kdeplot(data[:,metric_id+6], bw_method=0.5, ax=axins, color = "red",  linewidth=1.2,shade=False)
    #     # 调整子坐标系的显示范围
    #     axins.set_ylim(0,0.4)
    #     axins.set_xlim(-2,8)
    #     axins.set_xlabel(str(metric_name)+" values",fontdict=font2) #横轴名称
    #     axins.set_ylabel("Density",fontdict=font2) #纵轴名称
    # elif metric_name=="RMSE":
    #     # 绘制子图
    #     axins = ax.inset_axes((0.38, 0.25, 0.45, 0.45))
    #     cpc_gm_ind = sns.kdeplot(data[:,metric_id],  bw_method=0.5, ax=axins, color = "royalblue",linewidth=1.2,shade=False)
    #     cpc_rm_ind = sns.kdeplot(data[:,metric_id+3], bw_method=0.5, ax=axins, color = "green",  linewidth=1.2,shade=False)
    #     cpc_ggan_ind = sns.kdeplot(data[:,metric_id+6], bw_method=0.5, ax=axins, color = "red",  linewidth=1.2,shade=False)
    #     # 调整子坐标系的显示范围
    #     axins.set_ylim(0,0.05)
    #     axins.set_xlim(-40,50)
    #     axins.set_xlabel(str(metric_name)+" values",fontdict=font2) #横轴名称
    #     axins.set_ylabel("Density",fontdict=font2) #纵轴名称
    
    plt.savefig("./Figure/Density/"+str(region)+"_CPC_error.png",dpi=360,bbox_inches = 'tight')
    
    
#%%
def draw_bar(metric_id,metric_name,index, subplot_labels,data_values,subplot_values,region):
    font2 = {'family' : 'Arial','color':'Black','size': 12}
    font1 = {'family' : 'Arial','color':'Black','size': 18}
    width = 0.2
    # hatch_list = ['//', 'o', 'x', '.', '*', r'\\'] #设置线形
    #color_list = ['orange','royalblue','forestgreen']
    # color_list = ['#5f62b7','#f2a22c'] #红：#e02d4c  绿：#aadb3d
    color_list = ['royalblue','#aadb3d','orange','#e02d4c']
    # color_list = [xkcd_rgb['water blue'], xkcd_rgb['azure'], xkcd_rgb['bright sky blue'], xkcd_rgb['sage'], xkcd_rgb['sickly yellow'], xkcd_rgb['butterscotch']]
    plt.rcParams['xtick.direction'] = 'in' #坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in' #坐标轴刻度朝内
    fig = plt.figure(figsize=(8, 6))
    x_pad_width= width #0.2 # 分组柱状图中两个组之间的距离
    ax = fig.add_subplot(1, 1, 1)
    data = subplot_values
    ax.tick_params(axis='x',length =4,width=1, which='major',bottom='on') #,top=True
    ax.tick_params(axis='x',length =4, width=1, which='major',top='on')#,right=True
    ax.tick_params(axis='y',length =8,width=1, which='major', left='on') #,top=True
    ax.tick_params(axis='y',length =4,width=1, which='major', right='on') #,right=True
    ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(10))           #y轴次刻度间隔
    #ax.yaxis.grid(True, which='minor', linestyle='--', c='k')   #y轴次刻度虚线 
    #设置横纵坐标轴名称
    # ax.set_xlabel('County', fontdict=font1, labelpad=2)       #设置横刻度轴名称大小
    ax.set_ylabel(str(metric_name),fontdict=font1, labelpad=2)                      #设置纵刻度轴名称大小
    ax.set_xticks(np.arange(width, len(data)-width, 1))
    ax.set_xticklabels(index)
    # 设置刻度线标识的字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontsize(13) for label in labels]
    
    x_tick = np.arange(len(data)) # 标记三组数据的位置
    for x in range(len(data[0])): # 每次画n个bar
        height = data[:,x] # 是一个列表，有三个数据
        # print(height)
        label = subplot_labels[x]
        ax.bar(x_tick + x*x_pad_width, height=height, width=width, color=color_list[x],    # 同时画三个bar
               label=label,  align="center",edgecolor='black') #hatch=hatch_list[x],
        plt.rcParams.update({'font.size':13})
        ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.99, 0.99)) # 增加图例位置 bbox_to_anchor=(0, 1.)
    
    if metric_name=="RMSE":
        # 绘制子图
        # axins = ax.inset_axes((0.5, 0.25, 0.45, 0.45))
        axins = ax.inset_axes((0.25, 0.25, 0.7, 0.45))
        for x in range(len(data[0])): # 每次画n个bar
            height = data[:,x] # 是一个列表，有三个数据
            label = subplot_labels[x]
            # print(label)
            axins.bar(x_tick + x*x_pad_width, height=height, width=width, color=color_list[x],    # 同时画三个bar
                    label=label,  align="edge",edgecolor='black') #hatch=hatch_list[x],
            axins.set_ylabel("RMSE",fontdict=font2) #纵轴名称
            axins.set_xticks(np.arange(width, len(data)-width, 1))
            axins.set_xticklabels(index)
            axins.tick_params(axis='x',length =1,width=1, which='major',bottom='on') #,top=True
            axins.tick_params(axis='x',length =1, width=1, which='major',top='on')#,right=True
            axins.tick_params(axis='y',length =1,width=1, which='major', left='on') #,top=True
            axins.tick_params(axis='y',length =1,width=1, which='major', right='on') #,right=True
        # 调整子坐标系的显示范围
        # axins.set_ylim(0,20)
        # axins.set_xlim(7,12)
        
        axins.set_ylim(0,3.2)
        axins.set_xlim(2,12)
        
    if metric_name=="MAE":
        # 绘制子图
        # axins = ax.inset_axes((0.5, 0.25, 0.45, 0.45))
        axins = ax.inset_axes((0.25, 0.25, 0.7, 0.45))
        for x in range(len(data[0])): # 每次画n个bar
            height = data[:,x] # 是一个列表，有三个数据
            label = subplot_labels[x]
            # print(label)
            axins.bar(x_tick + x*x_pad_width, height=height, width=width, color=color_list[x],    # 同时画三个bar
                    label=label,  align="edge",edgecolor='black') #hatch=hatch_list[x],
            axins.set_ylabel("MAE",fontdict=font2) #纵轴名称
            axins.set_xticks(np.arange(width, len(data)-width, 1))
            axins.set_xticklabels(index)
            axins.tick_params(axis='x',length =1,width=1, which='major',bottom='on') #,top=True
            axins.tick_params(axis='x',length =1, width=1, which='major',top='on')#,right=True
            axins.tick_params(axis='y',length =1,width=1, which='major', left='on') #,top=True
            axins.tick_params(axis='y',length =1,width=1, which='major', right='on') #,right=True
        # 调整子坐标系的显示范围
        # axins.set_ylim(0,2.5)
        # axins.set_xlim(6,12)
            
        axins.set_ylim(0,1.2)
        axins.set_xlim(2,12)
        
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.35, hspace=0.6)
    plt.savefig("./Figure/Bar/Community_error_"+str(region)+"_"+str(metric_name)+".png", dpi=360, bbox_inches='tight')
    plt.show()
    
#%%
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.histplot(data='''    [ 1D data here ]    ''',
#              x='''    [ 1D data here ( group by what ) ]    ''',
#              binwidth='''    [int] interval length    ''', binrange='''    (min,max)    ''',
#              element='poly', fill=False, color='''    (r,g,b)    ''',label='''    legend name    ''',ax=ax)
# plt.legend()
    
#%%
def draw_line(metric_id,metric_name,index, subplot_labels,data_values,subplot_values,region):
    font1 = {'family' : 'Arial','color':'Black','size': 18}
    # font2 = {'family' : 'Arial','color':'Black','size': 12}
    width = 0
    color_list = ['royalblue','#aadb3d','#e02d4c']
    # color_list = [xkcd_rgb['water blue'], xkcd_rgb['azure'], xkcd_rgb['bright sky blue'], xkcd_rgb['sage'], xkcd_rgb['sickly yellow'], xkcd_rgb['butterscotch']]
    plt.rcParams['xtick.direction'] = 'in' #坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in' #坐标轴刻度朝内
    fig = plt.figure(figsize=(8, 6))
    x_pad_width=0.25 # 分组柱状图中两个组之间的距离
    ax = fig.add_subplot(1, 1, 1)
    data = subplot_values
    ax.tick_params(axis='x',length =4,width=1, which='major',bottom='on') #,top=True
    ax.tick_params(axis='x',length =4, width=1, which='major',top='on')#,right=True
    ax.tick_params(axis='y',length =8,width=1, which='major', left='on') #,top=True
    ax.tick_params(axis='y',length =4,width=1, which='major', right='on') #,right=True
    ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(10))           #y轴次刻度间隔
    #ax.yaxis.grid(True, which='minor', linestyle='--', c='k')   #y轴次刻度虚线 
    #设置横纵坐标轴名称
    ax.set_xlabel('Groups', fontdict=font1, labelpad=2)       #设置横刻度轴名称大小
    ax.set_ylabel('Average CPC',fontdict=font1, labelpad=2)              #设置纵刻度轴名称大小
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels(index)
    ax.set_xlim(0.5,10.5)
    # 设置刻度线标识的字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontsize(16) for label in labels]
    m_size = 8
    plt.plot(index,data_values[:,0],linewidth=1.5,c='royalblue',label="GM",marker='s',markersize=m_size,markeredgecolor='black',markeredgewidth=0.5)
    plt.plot(index,data_values[:,1],linewidth=1.5,c='#00aa00', label="RM",marker='^',markersize=m_size,markeredgecolor='black',markeredgewidth=0.5)
    plt.plot(index,data_values[:,2],linewidth=1.5,c='orange', label="PWO",marker='D',markersize=m_size,markeredgecolor='black',markeredgewidth=0.5)
    plt.plot(index,data_values[:,3],linewidth=1.5,c='#e02d4c', label="GGAN",marker='o',markersize=m_size,markeredgecolor='black',markeredgewidth=0.5)
    
    plt.rcParams.update({'font.size':13})
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.99, 0.99))
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.35, hspace=0.6)
    plt.savefig("./Figure/Line/Community_error_"+str(metric_name)+".png", dpi=360, bbox_inches='tight')
    plt.show()


#%%
def draw_density_data(data1,data2,region):
    import seaborn as sns 
    font2 = {'family' : 'Arial','size': 13}
    sns.set()  #切换到sns的默认运行配置
    sns.set_style(style='white')
    #设置图片的大小
    figsize = 8,5
    plt.rcParams['xtick.direction'] = 'in' ####坐标轴刻度朝内
    plt.rcParams['ytick.direction'] = 'in' ####坐标轴刻度朝内
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='x',length =4,width=1, which='major',bottom='on') #,top=True
    ax.tick_params(axis='x',length =4, width=1, which='major',top='on')#,right=True
    ax.tick_params(axis='y',length =8,width=1, which='major', left='on') #,top=True
    ax.tick_params(axis='y',length =4,width=1, which='major', right='on') #,right=True
    ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1);####设置上部坐标轴的粗细
    ax.set_yscale("log")
    # ax.set_xscale("log")
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    plt.ylabel("Frequency",fontdict=font1) #横轴名称
    plt.xlabel("Distance (km)",fontdict=font1) #纵轴名称
    plt.xticks(fontsize=14) #横轴刻度字体大小
    plt.yticks(fontsize=14) #纵轴刻度字体大小
    # ['royalblue','orange','#aadb3d']
    # if region=="bsh":
    plt.hist(data2, density=False, bins=100,label = "Inter-county",color = "royalblue",alpha=0.8)
    plt.hist(data1, density=False, bins=100,label = "Intra-county",color = "green",alpha=0.9)
    # elif region=="wdh":
    #     plt.hist(data2, density=False, bins=60,label = "Inter-county",color = "green",alpha=0.75)
    #     plt.hist(data1, density=False, bins=60,label = "Intra-county",color = "orange",alpha=0.75)
    # else:
    #     plt.hist(data2, density=False, bins=60,label = "Inter-county",color = "green",alpha=0.75)
    #     plt.hist(data1, density=False, bins=60,label = "Intra-county",color = "orange",alpha=0.75)
    # plt.rcParams.update({'font.size':15})
    
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.99, 0.99),prop=font2)
    # cpc_gm_ind = sns.distplot(data[:,metric_id],  label = "GM", ax=ax, color = "royalblue")
    # cpc_rm_ind = sns.distplot(data[:,metric_id+3], label = "RM", ax=ax, color = "green")
    # cpc_ggan_ind = sns.distplot(data[:,metric_id+6], label = "GGAN", ax=ax, color = "red")
    # plt.legend(loc="upper right")
    # if region=='bsh':
    ax.set_ylim(1,1000000)
    # if region=='uk':
    #     ax.set_ylim(0,24)
    # if region=='wdh':
    #     ax.set_ylim(0,18)
    
    plt.savefig("./Figure/"+str(region)+"_dis_distribution.png",dpi=360,bbox_inches = 'tight')
    