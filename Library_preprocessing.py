#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2023/11/21

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""
## read library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def make_dirs(path):
    import os
    try:
        os.makedirs(path)
    except:
        pass

def read_Raw_data(uploaded_file, index_col, header, DataBase, SAMPLE_INFO):
    try:
        raw_comp_data = pd.read_excel(uploaded_file, index_col=index_col, header=header)
    except:
        raw_comp_data = pd.read_csv(uploaded_file, index_col=index_col, header=header)

    raw_comp_data["DataBase"]=DataBase
    raw_comp_data["SAMPLE_INFO"]=SAMPLE_INFO

    return raw_comp_data



def Preprocessing_all(raw_data):

    ########################################## major data compile wt%->ppmの計算
    #DATA for exchange % to ppm
    elem_weight = pd.read_excel("List/Element_weight.xlsx", index_col=0).drop("O").T
    elem_list = elem_weight.columns
    word_list = elem_weight.T["Word"]
    ###### data save
    raw_data_save = pd.DataFrame(index = raw_data.index, columns=elem_list)
    for elem in elem_list:
        try: ## データがあるときは、raw_data_saveに保存
            elem_data_now = raw_data[elem].dropna()
            raw_data_save[elem].loc[elem_data_now.index] = elem_data_now
        except:
            pass

    ###### major data
    major_ppm = pd.DataFrame()
    for elem, word in zip(elem_list, word_list):
        try:
            # wt%->ppmの計算
            major_ppm[elem] = raw_data[word]*((10)**4)/elem_weight[elem]["%"]
        except:
            pass
    #raw_data assign; raw dataに代入
    raw_data[major_ppm.columns] = major_ppm
    #filled data assign; 既に入っているデータを再代入
    for elem in elem_list:
        raw_data_save_now = raw_data_save[elem].dropna()
        raw_data[elem].loc[raw_data_save_now.index] = raw_data_save_now
    ########################################## major data compile wt%->ppmの計算

    ########################################## CIA calculation
    raw_data["CaO*"] = raw_data["CaO"] - raw_data["P2O5"]/141.944/2*3/5
    raw_data["CIA*"] = 100*(raw_data["Al2O3"]/101.96)/((raw_data["Al2O3"]/101.96)+(raw_data["CaO"]/56.0774)+(raw_data["Na2O"]/61.9789)+(raw_data["K2O"]/94.2))
    raw_data["CIA"] = 100*(raw_data["Al2O3"]/101.96)/((raw_data["Al2O3"]/101.96)+(raw_data["CaO*"]/56.0774)+(raw_data["Na2O"]/61.9789)+(raw_data["K2O"]/94.2))
    ########################################## CIA calculation

    ########################################## Hughes (1972) diagram
    raw_data["K2O+Na2O"] = raw_data.K2O+raw_data.Na2O
    raw_data["100*K2O/(K2O+Na2O)"] = 100*raw_data.K2O/(raw_data.Na2O+raw_data.K2O)

    '''
    ##### now dont run
    sns.relplot(data=raw_data, x="100*K2O/(K2O+Na2O)", y="K2O+Na2O", hue='CIA', hue_order='CIA', aspect=1.4, )
    plt.plot([0, 15, 20, 30, 38], [2.5, 4, 5, 8.5, 13.5], color='yellow')
    plt.plot([0, 20, 80, 85], [0.8, 2, 6.5, 13.5], color='yellow')
    plt.ylim(0, 15)
    plt.title('Hughes (1972) diagram')
    plt.savefig('../Figure/Hughes (1972) diagram.pdf')
    plt.close()
    plt.show()
    '''
    ########################################## Hughes (1972) diagram

    ########################################## Primitive mantle normalize
    Primitive_not_applied = raw_data.copy()
    #normalizeのためのデータをread
    for_normalize_data = pd.read_excel("List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0)
    for_normalize_data = for_normalize_data.drop(columns=['Unnamed: 1']).T
    #一部のデータはErrorになるので，先にdrop
    for_normalize_data = for_normalize_data.drop(['F', 'In', 'Cl', 'Ge'])
    normalize_data_list = for_normalize_data.columns
    for_normalize_element = for_normalize_data.index.values
    for_normalize_element = np.append(for_normalize_element, "LOI")

    ## データの整理

    ## NaNを取り除く
    for elem in for_normalize_element:
        try:
            test = Primitive_not_applied[elem].astype(str)
            print(test.str.contains('NaN').value_counts())
            list__ = test[test.str.contains('NaN') ==True].index
            Primitive_not_applied.loc[list__, elem] = np.nan
        except:
            print("NOT_EXIST : " + elem)

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
    ## NANをdrop
    for elem in for_normalize_element:
        try:
            test = Primitive_not_applied[elem].astype(str)
            print(test.str.contains('NaN').value_counts())
            list__ = test[test.str.contains('NaN') ==True].index
            Primitive_not_applied.loc[list__, elem] = np.nan
        except:
            print("NOT_EXIST : " + elem)

    ## データの整理
    ## 一部object型になっているデータをfloat型に修正
    for col in Primitive_not_applied.columns:
        try:
            Primitive_not_applied[col] = Primitive_not_applied[col].astype("float")
        except:
            pass
    print("######################info")
    print(Primitive_not_applied.info())
    ## 一部object型になっているデータをfloat型に修正

    ## whole rockのデータのみを抽出
    if "ANALYZED MATERIAL" in Primitive_not_applied.columns:
        #Primitive_not_applied.fillna('WHOLE ROCK', inplace=True)
        #groupby インスタンスをread
        class_groupby = Primitive_not_applied.groupby("ANALYZED MATERIAL")
        #groupbyでWhole_rockのみread
        Primitive_not_applied_Whole_rock = class_groupby.get_group('WHOLE ROCK')
        #重複しているデータをdrop
        Primitive_not_applied_Whole_rock = Primitive_not_applied_Whole_rock.drop_duplicates()
    #共通するcolumns
    else:
        Primitive_not_applied_Whole_rock = Primitive_not_applied.copy()
    print(Primitive_not_applied_Whole_rock.shape)
    both_elem = list(set(for_normalize_data.index) & set(Primitive_not_applied.columns))

    #####################################ノーマライズするためのデータをread
    num = 2
    normalize_Primitive_name = normalize_data_list[num]
    num = 3
    normalize_Chondrite_name = normalize_data_list[num]
    #いらないelemを落とす
    for_normalize_data_compile = for_normalize_data[[normalize_Primitive_name, normalize_Chondrite_name]]
    for_normalize_data_compile = for_normalize_data_compile.dropna()
    #ノーマライズvalueをread
    Primitive_mantle_value = for_normalize_data_compile[normalize_Primitive_name]
    C1_chondrite = for_normalize_data_compile[normalize_Chondrite_name]
    ###ノーマライズする前のデータ
    Whole_rock_before_Normalize = Primitive_not_applied_Whole_rock[both_elem]
    ###ノーマライズしたのデータ
    Whole_rock_after_Normalize_PM = pd.DataFrame()
    Whole_rock_after_Normalize_C1 = pd.DataFrame()
    ###ノーマライズ出来ないデータ
    Whole_rock_cannot_Normalize = Primitive_not_applied_Whole_rock.drop(both_elem , axis = 1)
    ##ノーマライズ / 行列計算
    Whole_rock_after_Normalize_PM = Whole_rock_before_Normalize/for_normalize_data['PM(SM89)']
    Whole_rock_after_Normalize_C1 = Whole_rock_before_Normalize/for_normalize_data['CI(SM89)']
    ########################################## Primitive mantle normalize
    return  Primitive_not_applied_Whole_rock, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1

def save_preprocessed_data(path_name, data_name, Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1):
    path_name = path_name+data_name+'/'
    make_dirs(path_name)
    ###output
    try:
        Location_Ref_Data = Whole_rock_cannot_Normalize[["DataBase", "SAMPLE_INFO", 'LATITUDE', 'LONGITUDE']]
    except:
        Location_Ref_Data = Whole_rock_cannot_Normalize[["DataBase", "SAMPLE_INFO"]]

    #compile
    PM = pd.concat([Whole_rock_after_Normalize_PM, Whole_rock_cannot_Normalize], axis = 1)
    C1 = pd.concat([Whole_rock_after_Normalize_C1, Whole_rock_cannot_Normalize], axis = 1)

    #基本ファイル
    Whole_rock_RAW.to_excel(path_name + "Primitive_not_applied.xlsx")
    #only normalized
    Whole_rock_after_Normalize_PM.to_excel(path_name + "PM_applied.xlsx")
    Whole_rock_after_Normalize_C1.to_excel(path_name + "C1_applied.xlsx")
    #compile_normalized
    PM.to_excel(path_name + "PM_applied_with.xlsx")
    C1.to_excel(path_name + "C1_applied_with.xlsx")
    Whole_rock_cannot_Normalize.to_excel(path_name + "Cannot_applied.xlsx")
    Location_Ref_Data.to_excel(path_name + "Location_Ref_Data.xlsx")
    return 0
