#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2026/02/08

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def read_Raw_data(uploaded_file, index_col, header, DataBase, SAMPLE_INFO):
    try:
        raw_comp_data = pd.read_excel(uploaded_file, index_col=index_col, header=header)
    except:
        raw_comp_data = pd.read_csv(uploaded_file, index_col=index_col, header=header)

    raw_comp_data["DataBase"]=DataBase
    raw_comp_data["SAMPLE_INFO"]=SAMPLE_INFO

    return raw_comp_data

def exchange_major_to_ppm(Data_bef):
    elem_weight = pd.read_excel("List/Element_weight.xlsx", index_col=0).drop("O").T
    elem_list = elem_weight.columns
    word_list = elem_weight.T["Word"]

    ######もうすでにある場合のデータを保存
    already_exist_data = pd.DataFrame(index = Data_bef.index, columns=elem_list)

    for elem in elem_list:
        #指定した元素で既にppmで計算されているデータをalready_exist_dataに保存
        try:
            now_data = Data_bef[elem].dropna()
            already_exist_data[elem].loc[now_data.index] = now_data
        except:
            ##### 指定した元素がない場合はpass
            pass

    #入れる箱を準備
    major_ppm = pd.DataFrame()
    for elem, word in zip(elem_list, word_list):
        try:
            # %　-＞　ppmの計算
            major_ppm[elem] = Data_bef[word]*((10)**4)/elem_weight[elem]["%"]
        except:
            pass
            #word_list =word_list.drop(elem)
            #print(elem)
            #print(word_list)

    #Data_befに代入
    Data_bef[major_ppm.columns] = major_ppm

    ######もうすでにある場合のデータを保存したものをf
    ######再代入
    for elem in elem_list:

        try:
            now_data = already_exist_data[elem].dropna()
            Data_bef[elem].loc[now_data.index] = now_data
        except:
            pass
    return Data_bef

def CIA_value_calc(Data_bef):
    try:
        Data_bef["CaO*"] = Data_bef["CaO"] - Data_bef["P2O5"]/141.944/2*3/5
        Data_bef["CIA*"] = 100*(Data_bef["Al2O3"]/101.96)/((Data_bef["Al2O3"]/101.96)+(Data_bef["CaO"]/56.0774)+(Data_bef["Na2O"]/61.9789)+(Data_bef["K2O"]/94.2))
        Data_bef["CIA"] = 100*(Data_bef["Al2O3"]/101.96)/((Data_bef["Al2O3"]/101.96)+(Data_bef["CaO*"]/56.0774)+(Data_bef["Na2O"]/61.9789)+(Data_bef["K2O"]/94.2))
    except:
        pass

    try:
        if st.checkbox('Show CIA distribution'):

            fig = px.histogram(Data_bef, x="CIA")
            st.plotly_chart(fig)
    except:
        st.write("This data cannnot calculate CIA")
    return Data_bef


def primitive_applied(Primitive_not_applied):
    #normalizeのためのデータをread
    for_normalize_data = pd.read_excel("List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0)
    for_normalize_data = for_normalize_data.drop(columns=['Unnamed: 1']).T

    #一部のデータはErrorになるので，先にdrop
    for_normalize_data = for_normalize_data.drop(['F', 'In', 'Cl', 'Ge'])
    normalize_data_list = for_normalize_data.columns
    for_normalize_element = for_normalize_data.index.values
    for_normalize_element = np.append(for_normalize_element, "LOI")

    ################################################## データの整理
    ##数値 < 0 をdrop
    for elem in for_normalize_element:
        try:
            test = Primitive_not_applied[elem].astype(str)
            print(test.str.contains('<').value_counts())
            list__ = test[test.str.contains('<') ==True].index
            Primitive_not_applied.loc[list__, elem] = np.nan
        except:
            print("NOT_EXIST : " + elem)

    for elem in for_normalize_element:
        try:
            test = Primitive_not_applied[elem].astype(str)
            print(test.str.contains('>').value_counts())
            list__ = test[test.str.contains('>') ==True].index
            Primitive_not_applied.loc[list__, elem] = np.nan
        except:
            print("NOT_EXIST : " + elem)

    ## 数値 == 0 をdrop
    for elem in for_normalize_element:
        try:
            test = Primitive_not_applied[elem].astype(float)
            print(test[test==0].value_counts())
            list__ = test[test==0].index
            Primitive_not_applied.loc[list__, elem] = np.nan
        except:
            print("NOT_EXIST : " + elem)

    ## 数値 < 0 をdrop
    for elem in for_normalize_element:
        try:
            test = Primitive_not_applied[elem].astype(float)
            print(test[test<0].value_counts())
            list__ = test[test<0].index
            Primitive_not_applied.loc[list__, elem] = np.nan
        except:
            print("NOT_EXIST : " + elem)
    ################################################## データの整理
    ################################################## 一部object型になっているデータをfloat型に修正
    for col in Primitive_not_applied.columns:
        try:
            Primitive_not_applied[col] = Primitive_not_applied[col].astype("float")
        except:
            pass
    print("######################info")
    print(Primitive_not_applied.info())
    ################################################## 一部object型になっているデータをfloat型に修正

    ################################################## Normalize
    #代入
    Whole_rock = Primitive_not_applied
    print(Whole_rock.shape)
    #共通するcolumns
    both_elem = list(set(for_normalize_data.index) & set(Primitive_not_applied.columns))

    #####################################ノーマライズするためのデータをread
    num = 2
    normalize_Primitive_name = normalize_data_list[num]
    print(normalize_Primitive_name)
    num = 3
    normalize_Chondrite_name = normalize_data_list[num]
    print(normalize_Chondrite_name)

    #いらないelemを落とす
    for_normalize_data_compile = for_normalize_data[[normalize_Primitive_name, normalize_Chondrite_name]]
    for_normalize_data_compile = for_normalize_data_compile.dropna()

    #ノーマライズvalueをread
    Primitive_mantle_value = for_normalize_data_compile[normalize_Primitive_name]
    C1_chondrite = for_normalize_data_compile[normalize_Chondrite_name]

    #####################################ノーマライズするデータをread
    ###ノーマライズする前のデータ
    Whole_rock_before_Normalize = Whole_rock[both_elem]
    ###ノーマライズしたのデータ
    Whole_rock_after_Normalize_PM = pd.DataFrame()
    Whole_rock_after_Normalize_C1 = pd.DataFrame()
    ###ノーマライズ出来ないデータ
    Whole_rock_cannot_Normalize = Whole_rock.drop(both_elem , axis = 1)

    #####################################ノーマライズ / 行列計算
    Whole_rock_after_Normalize_PM = Whole_rock_before_Normalize/for_normalize_data['PM(SM89)']
    Whole_rock_after_Normalize_C1 = Whole_rock_before_Normalize/for_normalize_data['CI(SM89)']
    ################################################## Normalize

    return Whole_rock, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1


def preprocessing_normalize_output(data, DataBase, SAMPLE_INFO, location_info):

    data_all = data.copy()
    #データの準備
    Data_bef = data_all.copy()
    ##################################################### major DATA for exchange % to ppm
    Data_bef = exchange_major_to_ppm(Data_bef)
    ##################################################### Calc CIA
    #Data_bef=CIA_value_calc(Data_bef)
    ##################################################### Normalize data
    Primitive_not_applied = Data_bef.copy()
    Whole_rock, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1 = primitive_applied(Primitive_not_applied)

    ##################################################### Save
    #compile
    PM = pd.concat([Whole_rock_after_Normalize_PM, Whole_rock_cannot_Normalize], axis = 1)
    C1 = pd.concat([Whole_rock_after_Normalize_C1, Whole_rock_cannot_Normalize], axis = 1)
    Location_Ref_Data = Whole_rock_cannot_Normalize[location_info]

    st.subheader("Download preprocessed data")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="Primitive mantle (SM89) download",
            data=PM.to_csv().encode('utf-8'),
            file_name="PM_applied_with.csv",
            mime='text/csv',
            )
    with col2:
        st.download_button(
            label="C1 chondrite (SM89) download",
            data=C1.to_csv().encode('utf-8'),
            file_name="C1_applied_with.csv",
            mime='text/csv',
            )
    with col3:
        st.download_button(
            label="Cannot_Normalize download",
            data=Whole_rock_cannot_Normalize.to_csv().encode('utf-8'),
            file_name="Cannot_applied.csv",
            mime='text/csv',
            )
    with col4:
        st.download_button(
            label="Location_Ref_Data download",
            data=Location_Ref_Data.to_csv().encode('utf-8'),
            file_name="Location_Ref_Data.csv",
            mime='text/csv',
            )


    return PM, Location_Ref_Data




###########################################################################
###########################################################################
###########################################################################

# Visualization

###########################################################################
###########################################################################
###########################################################################

def Spidergram_simple(data, scale, legend, color, style, label, fig, ax):
    ######## Define X(columns) and X length
    columns = data.columns
    length = data.values.shape[1]
    t = np.arange(0,length, 1)
    ######## Define X(columns) and X length




    ######## Define cycle of plotting
    sample_nun = len(data.index)
    ######## Define X axis and label
    plt.xticks(t, columns)
    plt.xticks(rotation = 70)

    if(scale == "log"):
        plt.yscale("log")

    for num in range(sample_nun):
        if num ==0:
            plt.errorbar(t, data.values[num], color = color, linestyle=style, label = label, linewidth = 3.0)
        else:
            plt.errorbar(t, data.values[num], color = color, linestyle=style, linewidth = 3.0)

    if (legend == "on"):
        plt.legend(data.index)
    return fig

def Spidergram_marker(data, immobile_elem, color, markeredgecolor, style, size, fig, ax):
    ######## Define X(columns) and X length
    columns = data.columns
    length = data.values.shape[1]
    t = np.arange(0,length, 1)
    ######## Define X(columns) and X length

    ######## define marker elem
    marker_row = []
    for immobile_elem_now in immobile_elem:
        num = list(columns.values).index(immobile_elem_now)
        marker_row.append(num)
    marker_row.sort() # compare with t

    ######## Define cycle of plotting
    sample_nun = len(data.index)
    ######## Define cycle of plotting

    ######## marker plot
    for num in range(sample_nun):
        plt.errorbar(t, data.values[num], color = color, linestyle="None", marker=style, \
                     markersize=size, markevery=marker_row, markeredgecolor=markeredgecolor, markeredgewidth=2.5)
    ######## marker plot

    return fig

def Spidergram_error(data, model_score,scale, legend, color, style, label, fig, ax):
    ######## Define X(columns) and X length
    columns = data.columns
    length = data.values.shape[1]
    t = np.arange(0,length, 1)
    sample_nun = len(data.index)
    data=data.astype(float)
    ######## Define X(columns) and X length

    ################################# To plot Error bar
    #score_elem
    model_score = model_score[columns]

    #error_bar_calculation log_scale
    error_plus_log = data.apply(lambda x : np.log10(x)) + model_score
    error_minus_log = data.apply(lambda x : np.log10(x)) - model_score
    raw_log = data.apply(lambda x : np.log10(x)).values
    #error_bar_calculation normal_scale
    error_plus = error_plus_log.apply(lambda x : 10**x) - 10**raw_log
    error_minus = 10**raw_log - error_minus_log.apply(lambda x : 10**x)
    error = pd.concat([error_minus, error_plus], axis = 0).values
    ################################# To plot Error bar

    ######## Define X axis and label
    plt.xticks(t, columns)
    plt.xticks(rotation = 70)
    if(scale == "log"):
        plt.yscale("log")

    for num in range(sample_nun):
        if num ==0:
            plt.errorbar(t, data.values[num], yerr = error, color = color, linestyle=style, label = label, capsize=5, linewidth = 3.0)
        else:
            plt.errorbar(t, data.values[num], yerr = error, color = color, linestyle=style, capsize=5, linewidth = 3.0)

    if (legend == "on"):
        plt.legend(data.index)
    return fig

def Spidergram_fill_between(data, scale, legend, color, style, label, alpha, fig, ax):

    ######## Define X(columns) and X length
    columns = data.columns
    length = data.values.shape[1]
    t = np.arange(0,length, 1)
    ######## Define X(columns) and X length

    ######## Define MinMax
    # Min Maxを求めて， 上限下限の値を決める
    data_max = pd.DataFrame(data.max()).T
    data_min = pd.DataFrame(data.min()).T
    ######## Define MinMax

    ######## Define cycle of plotting
    sample_nun = len(data.index)
    ######## Define X axis and label
    plt.xticks(t, columns)
    plt.xticks(rotation = 70)
    if(scale == "log"):
        plt.yscale("log")

    plt.fill_between(t, data_max.values[0], data_min.values[0], color = color, linestyle=style, label = label, alpha = alpha)

    if (legend == "on"):
        plt.legend(data.index)

def Spidergram_fill_immobile(columns, immobile_elem, color, alpha, fig, ax):
    ######## Define X(columns) and X length
    length = len(columns)
    t = np.arange(0,length, 1)
    ######## Define X(columns) and X length

    ######## Define X axis and label
    plt.xticks(t, columns)
    plt.xticks(rotation = 70)
    ######## Define X axis and label

    ######## Define fill index and fill X axis
    # 箱を準備
    elem_fill_x = []
    for elem in immobile_elem:
        # 一致しているインデックス番号を代入
        #elem_fill_x.append(list(columns).index(elem))
        num = list(columns).index(elem)
        plt.axvspan(num-0.45, num+0.45, color = color, alpha = alpha)
    ######## Define fill index
    return fig

def Spidergram_ragne_as_error_bar(data, range, scale, legend, color, style, label, fig, ax):
    columns = data.columns
    length = data.values.shape[1]
    t = np.arange(0,length, 1)
    sample_nun = len(data.index)
    data=data.astype(float)
    ######## Define X(columns) and X length

    ################################# To plot Error bar
    #score_elem
    error_plus_log = (data*range[0]).apply(lambda x : np.log10(x))
    error_minus_log = (data*range[1]).apply(lambda x : np.log10(x))
    raw_log = data.apply(lambda x : np.log10(x)).values

    #error_bar_calculation normal_scale
    error_plus = error_plus_log.apply(lambda x : 10**x) - 10**raw_log
    error_minus = 10**raw_log - error_minus_log.apply(lambda x : 10**x)
    error = pd.concat([error_minus, error_plus], axis = 0)#values
    ################################# To plot Error bar

    ######## Define X axis and label
    plt.xticks(t, columns)
    plt.xticks(rotation = 70)
    if(scale == "log"):
        plt.yscale("log")
    plt.errorbar(t, data.values.T, yerr = error.values, color = color, linestyle=style, capsize=5, linewidth = 3.0)

def PM_to_ppm(data): # ver 260208
    now_data = data.copy()
    
    # 1. ファイル読み込み
    try:
        for_normalize_data = pd.read_excel("List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0)
        pm_values = for_normalize_data.loc["PM(SM89)"]
    except FileNotFoundError:
        print("【Error】Excel file not found. Check the path: 'List/Primitive_Mantle _ C1 Chondrite.xlsx'")
        return data
    except KeyError:
        print("【Error】'PM(SM89)' not found in the Excel file.")
        return data

    for elem in data.columns:
        # 2. "_" を消してPM値のリストにあるか確認
        elem_check = elem.replace("_", "")

        if elem_check not in pm_values.index:
            # ターミナルに警告を表示
            print(f"【Warning】'{elem}' was skipped: PM value for '{elem_check}' not found.")
            continue
        
        # 3. 計算実行
        value = pm_values[elem_check]
        now_data[elem] = now_data[elem] * value
            
    return now_data