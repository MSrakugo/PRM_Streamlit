#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2024/02/07

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""
#plot&calc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.ticker as ticker
import streamlit as st

#import Library_figure as PRM_figure

import scipy
import scipy.optimize as optimize
import sklearn.metrics as metrics
from sklearn.metrics import r2_score

#pickle
import pickle

#ファイル管理系
import os
import glob

def make_dirs(path):
    try:
        os.makedirs(path)
        print("Correctly make dirs")
    except:
        print("Already exist or fail to make dirs")
        
def std_mean(t):
    value_return = np.sqrt(np.average((t)**2))
    return value_return
def score_mean(t):
    value_return = np.average(t)
    return value_return

plt.rcParams["font.size"] = 25
plt.rcParams['font.sans-serif'] = ['Arial']

def Error_Distribution_Figure(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list):
    now_figure_path = path+'/Model_explain'
    make_dirs(now_figure_path)
    Raw_Protolith_location = pd.read_excel(path_model+"/Protolith/Location_Ref_Data.xlsx", index_col=0)
    #normalizeのためのデータをread
    for_normalize_data = pd.read_excel("List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0, header=0).loc['PM(SM89)']#[trace_all]
    normalized_element = for_normalize_data.dropna().index[1:]

    ################ fig setting
    # 表示するタイプ
    hist_family = ['count', 'frequency', 'density', 'probability']
    hist_family_label = ['Count', 'Frequency', 'Density', 'Probability']

    num_first = -1
    num = num_first

    # 表示の形を決める
    len_elem = len(mobile_elem_all)
    ncols = 3 # 選べる様にする
    nrows = len_elem//3 +1

    figsize_num = 9.5
    figsize=(ncols*figsize_num, nrows*figsize_num)
    font_size_label = 40
    ############### fig setting

    ################################################################ # hist name を変えてForループ
    for hist_name, hist_name_label in zip(hist_family, hist_family_label):
        #figureの形と数
        num = num + 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        ##################################figure location listの作成
        loc_list = []
        loc_rows = -1
        for num in range(len(mobile_elem_all)):
            loc_cols = num%ncols
            if loc_cols == 0:
                loc_rows=loc_rows+1
            loc_list.append([loc_rows, loc_cols])
        ##################################figure location listの作成

        ###########移動元素ごとのスコアを求める
        for loc_num, define_mobile_elem in enumerate(mobile_elem_all):
            # ax
            ax=axes[loc_list[loc_num][0]][loc_list[loc_num][1]]
            now_path = path+'/'+define_mobile_elem
            now_figure_path = path+'/Model_explain'
            make_dirs(now_figure_path)
            print(now_path)
            print(define_mobile_elem)
            if define_mobile_elem in normalized_element :
                data = pd.read_excel(now_path+'/test_data_all.xlsx', index_col=0) * for_normalize_data[define_mobile_elem] ## mobile elemでPM→ppmに戻す
            else:
                data = pd.read_excel(now_path+'/test_data_all.xlsx', index_col=0)
            data['Error'] = data['predict'] / data['RAW']
            data['SAMPLE_INFO'] = Raw_Protolith_location.loc[data.index]['SAMPLE_INFO']
            data['Tectonic setting'] = data['SAMPLE_INFO']

            # 相関係数を求める
            corr_each_elem = data[['RAW', 'predict']].apply(lambda x : np.log10(x)).corr().loc['RAW']['predict']

            ################################################################ Setting
            #元素によって範囲を変更 histgram
            if define_mobile_elem in good_range_elem:
                binrange_x = [0.1, 10]
                binrange_sns = [-1, 1]# log表記
                set_xticks = [0.1, 1.0, 10.0]
            else:
                binrange_x = [0.1, 10]
                binrange_sns = [-1, 1]# log表記
                set_xticks = [0.1, 1.0, 10.0]
            # only Rb display legend
            legend_flag = False
            if (define_mobile_elem == 'Rb')&(hist_name ==hist_family[0]) :
                legend_flag = True
            ################################################################ Setting
            ################################################################ PLOT
            print(data.shape)
            #plt.figure(figsize=(10, 10))

            for tec_setting in TECTONIC_list:
                index = data[data['Tectonic setting'] == tec_setting].index
                print(index.shape)
                sns.histplot(data = data.loc[index], x='Error', hue = 'Tectonic setting', log_scale=True, stat=hist_name,\
                            hue_order=TECTONIC_list, palette='Dark2', element='step', fill=False, linewidth=7, legend=legend_flag, bins=30, binrange=binrange_sns, ax=ax)

            # Setting
            ax.set_xlabel('Predicted / Raw', fontsize=font_size_label)
            ax.set_ylabel(hist_name_label, fontsize=font_size_label)
            ax.set_xlim(binrange_x)
            ax.set_title(define_mobile_elem, fontsize=font_size_label*1.2)
            ax.xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
            ax.yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis = 'x', labelsize=font_size_label/1.2)
            ax.tick_params(axis = 'y', labelsize=font_size_label/1.2)
            ax.tick_params(which = 'major', length = font_size_label/3, width = font_size_label/15)
            ax.tick_params(which = 'minor', length = font_size_label/6, width = font_size_label/30)
            ax.get_xaxis().set_tick_params(pad=font_size_label/2)
            ax.xaxis.set_ticks(set_xticks)
        plt.tight_layout()
        plt.savefig(now_figure_path+'/0_test_error_distribution_' + hist_name + '.pdf', bbox_inches='tight')
        st.pyplot(fig)
        plt.clf()
        plt.close()
    ######################



def Usual_Scatter_Plot(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list):
    now_figure_path = path+'/Model_explain'
    make_dirs(now_figure_path)
    Raw_Protolith_location = pd.read_excel(path_model+"/Protolith/Location_Ref_Data.xlsx", index_col=0)
    #normalizeのためのデータをread
    for_normalize_data = pd.read_excel("List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0, header=0).loc['PM(SM89)']#[trace_all]
    normalized_element = for_normalize_data.dropna().index[1:]

    ################ fig setting
    num_first = -1
    num = num_first

    # 表示の形を決める
    len_elem = len(mobile_elem_all)
    ncols = 3 # 選べる様にする
    nrows = len_elem//3 +1

    figsize_num = 9.5
    figsize=(ncols*figsize_num, nrows*figsize_num)
    font_size_label = 40
    ############### fig setting

    ###################################################################################### #scatter
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ### corr^2
    corr_compile = pd.DataFrame(index=['R2_score'], columns=mobile_elem_all)
    ##################################figure location listの作成
    loc_list = []
    loc_rows = -1
    for num in range(len(mobile_elem_all)):
        loc_cols = num%ncols
        if loc_cols == 0:
            loc_rows=loc_rows+1
        loc_list.append([loc_rows, loc_cols])
    ##################################figure location listの作成

    ###########移動元素ごとのスコアを求める
    for loc_num, define_mobile_elem in enumerate(mobile_elem_all):
        # ax
        ax=axes[loc_list[loc_num][0]][loc_list[loc_num][1]]
        now_path = path+'/'+define_mobile_elem
        now_figure_path = path+'/Model_explain'
        make_dirs(now_figure_path)
        print(now_path)
        print(define_mobile_elem)
        data = pd.read_excel(now_path+'/test_data_all.xlsx', index_col=0) 

        if define_mobile_elem in normalized_element:
            data[[define_mobile_elem, 'predict', 'RAW']] = data[[define_mobile_elem, 'predict', 'RAW']]* for_normalize_data[define_mobile_elem]## mobile elemでPM→ppmに戻す
        else:
            data[[define_mobile_elem, 'predict', 'RAW']] = data[[define_mobile_elem, 'predict', 'RAW']]## mobile elemでPM→ppmに戻す
        
        data['Error'] = data['predict'] / data['RAW']
        data['SAMPLE_INFO'] = Raw_Protolith_location.loc[data.index]['SAMPLE_INFO']
        data['Tectonic setting'] = data['SAMPLE_INFO']

        # 相関係数を求める
        corr_each_elem = r2_score(data['RAW'].apply(lambda x : np.log10(x)), data['predict'].apply(lambda x : np.log10(x)))
        corr_compile.loc['R2_score',define_mobile_elem]=corr_each_elem*corr_each_elem
        print(corr_compile.loc['R2_score',define_mobile_elem])
        ################################################################ Setting
        legend_flag = False
        if define_mobile_elem == 'Rb':
            #legend_flag = True
            legend_flag = False
        ################################################################ Setting

        ################################################################ # scatter plot
        min = np.log10(data['RAW'].min()) - 0.3
        max = np.log10(data['RAW'].max()) + 0.3
        # -5～5まで1刻みのデータを作成
        x = np.arange(10**min, 10**max)
        # 直線の式を定義
        y = x
        ax.plot(x,y, linestyle = ":", c = '#63b9af')

        sns.scatterplot(data=data, x='RAW', y='predict', hue='Tectonic setting',\
                           hue_order=TECTONIC_list, palette='Dark2', style='Tectonic setting', s=200, alpha=1, edgecolor="none", legend=legend_flag, ax=ax)
        ax.set_title(define_mobile_elem, fontsize=font_size_label*1.2)
        ax.set_xlabel('Raw [ppm]', fontsize=font_size_label)
        ax.set_ylabel('Predicted [ppm]', fontsize=font_size_label)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax.yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis = 'x', labelsize=font_size_label/1.2)
        ax.tick_params(axis = 'y', labelsize=font_size_label/1.2)
        ax.tick_params(which = 'major', length = font_size_label/3, width = font_size_label/15)
        ax.tick_params(which = 'minor', length = font_size_label/6, width = font_size_label/30)
        ax.get_xaxis().set_tick_params(pad=font_size_label/2)

        ax.text(0.95, 0.05, 'R={:.3f}'.format(corr_each_elem), horizontalalignment='right', transform=ax.transAxes, fontsize = 50)

    plt.tight_layout()
    plt.savefig(now_figure_path+'/0_raw_vs_predict.pdf', bbox_inches='tight')
    st.pyplot(fig)
    plt.clf()
    plt.close()
    ##output corr
    corr_compile.to_excel(now_figure_path+'/0_corr_compile.xlsx')
    corr_compile
    ######################################################################################## #scatter

def NGBoost_Scatter_Plot(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list):
    now_figure_path = path+'/Model_explain'
    make_dirs(now_figure_path)
    Raw_Protolith_location = pd.read_excel(path_model+"/Protolith/Location_Ref_Data.xlsx", index_col=0)
    #normalizeのためのデータをread
    for_normalize_data = pd.read_excel("List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0, header=0).loc['PM(SM89)']#[trace_all]
    normalized_element = for_normalize_data.dropna().index[1:]

    ################ fig setting
    num_first = -1
    num = num_first

    # 表示の形を決める
    len_elem = len(mobile_elem_all)
    ncols = 3 # 選べる様にする
    nrows = len_elem//3 +1

    figsize_num = 9.5
    figsize=(ncols*figsize_num, nrows*figsize_num)
    font_size_label = 40
    ############### fig setting

    ###################################################################################### #scatter
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ### corr^2
    corr_compile = pd.DataFrame(index=['R2_score'], columns=mobile_elem_all)
    ##################################figure location listの作成
    loc_list = []
    loc_rows = -1
    for num in range(len(mobile_elem_all)):
        loc_cols = num%ncols
        if loc_cols == 0:
            loc_rows=loc_rows+1
        loc_list.append([loc_rows, loc_cols])
    ##################################figure location listの作成

    ###########移動元素ごとのスコアを求める
    for loc_num, define_mobile_elem in enumerate(mobile_elem_all):
        # ax
        ax=axes[loc_list[loc_num][0]][loc_list[loc_num][1]]
        now_path = path+'/'+define_mobile_elem
        now_figure_path = path+'/Model_explain'
        make_dirs(now_figure_path)
        print(now_path)
        print(define_mobile_elem)
        data = pd.read_excel(now_path+'/test_data_all.xlsx', index_col=0)
        if define_mobile_elem in normalized_element:
            data[[define_mobile_elem, 'predict', 'RAW']] = data[[define_mobile_elem, 'predict', 'RAW']]* for_normalize_data[define_mobile_elem]## mobile elemでPM→ppmに戻す
        else:
            data[[define_mobile_elem, 'predict', 'RAW']] = data[[define_mobile_elem, 'predict', 'RAW']]## mobile elemでPM→ppmに戻す
        
        data['Error'] = data['predict'] / data['RAW']
        data['SAMPLE_INFO'] = Raw_Protolith_location.loc[data.index]['SAMPLE_INFO']
        data['Tectonic setting'] = data['SAMPLE_INFO']
        data['Tectonic setting_num'] = data['SAMPLE_INFO']
        data["RAW"] = data["RAW"].apply(lambda x : np.log10(x))
        data["predict"] = data["predict"].apply(lambda x : np.log10(x))
        ################################################################ Setting
        legend_flag = False
        if define_mobile_elem == 'Rb':
            #legend_flag = True
            legend_flag = False
        ################################################################ Setting

        ################################################################ # scatter plot
        min = data['RAW'].min()
        max = data['RAW'].max()
        # -5～5まで1刻みのデータを作成
        x = np.arange(min, max)
        # 直線の式を定義
        y = x
        ax.plot(x,y, linestyle = ":", c = '#63b9af')

        ax.errorbar(x=data["RAW"], y=data["predict"],\
                    yerr = data["predict_Dist"], capsize=5, fmt='o', markersize=1, ecolor='black', markeredgecolor = "black", alpha=0.3, zorder=1)
        sns.scatterplot(data=data, x='RAW', y='predict', hue='Tectonic setting',\
                           hue_order=TECTONIC_list, palette='Dark2', style='Tectonic setting', s=200, alpha=1, edgecolor="none", legend=legend_flag, ax=ax)

        ax.set_title(define_mobile_elem, fontsize=font_size_label*1.2)
        ax.set_xlabel('log10(Raw) [ppm]', fontsize=font_size_label)
        ax.set_ylabel('log10(Predicted) [ppm]', fontsize=font_size_label)
        ax.xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax.yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis = 'x', labelsize=font_size_label/1.2)
        ax.tick_params(axis = 'y', labelsize=font_size_label/1.2)
        ax.tick_params(which = 'major', length = font_size_label/3, width = font_size_label/15)
        ax.tick_params(which = 'minor', length = font_size_label/6, width = font_size_label/30)
        ax.get_xaxis().set_tick_params(pad=font_size_label/2)
    plt.tight_layout()
    plt.savefig(now_figure_path+'/0_raw_vs_predict_std_color.pdf', bbox_inches='tight')
    plt.savefig(now_figure_path+'/0_raw_vs_predict_std_color.jpg', bbox_inches='tight')
    st.pyplot(fig)
    plt.clf()
    plt.close()

