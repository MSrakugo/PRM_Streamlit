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
#plot&calc
import pandas as pd
import numpy as np

#統計処理
import itertools

#pickle
import pickle

#ファイル管理系
import os

#ランダム
import random
random.seed(0)

import streamlit as st

#model
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


################################################################# MODEL
#################################################クラスの定義
class Models:
    def __init__(self, model, sc, pca, ICA):
        self.model = model
        self.sc = sc
        self.pca = pca
        self.ICA = ICA

class Model_datas:
    def __init__(self, Score_all, test_error_all, test_data_all):
        self.Score_all = Score_all
        self.test_error_all = test_error_all
        self.test_data_all = test_data_all

class Model_feature_setting: # feature setting
    def __init__(self, ML_algorithm_name, setting_X_raw, setting_X_log, setting_X_product, setting_X_product_log, setting_X_ratio, setting_X_ratio_log, setting_NORMAL_OR_LOG, \
                 setting_PCA, setting_ICA, setting_standard_scaler):
        #Model name LightGBM, NGBoost
        self.ML_algorithm_name = ML_algorithm_name

        #Setting
        self.setting_X_raw = setting_X_raw
        self.setting_X_log = setting_X_log
        self.setting_X_product = setting_X_product
        self.setting_X_product_log = setting_X_product_log

        self.setting_X_ratio = setting_X_ratio
        self.setting_X_ratio_log = setting_X_ratio_log

        #standard_scalerに与える時，X_log or Xを選択
        self.setting_NORMAL_OR_LOG = setting_NORMAL_OR_LOG
        self.setting_PCA = setting_PCA
        self.setting_ICA = setting_ICA
        self.setting_standard_scaler = setting_standard_scaler

#################################################クラスの定義
#################################################計算関連
def make_dirs(path):
    try:
        os.makedirs(path)
        print("Correctly make dirs")
    except:
        print("Already exist or fail to make dirs")

### 元素組み合わせを作る関数###############################################################
def elem_combinations_list(columns):
    num = 0
    first_list = []
    secound_list = []
    for v in itertools.combinations(columns, 2):
        first_list.append(v[0])
        secound_list.append(v[1])
    return first_list, secound_list

def elem_ratio_list(data):
    ratio_data = pd.DataFrame()
    columns = data.columns
    first_list, secound_list = elem_combinations_list(columns)
    for elem1, elem2 in zip(first_list, secound_list):
        ratio_name_1 = elem1 + "/" + elem2
        ratio_name_2 = elem2 + "/" + elem1
        ratio_name = elem1 + "/" + elem2
        try:
            first_data = data[elem1]
            secound_data = data[elem2]
            ratio_value = first_data/secound_data
            ratio_data[ratio_name_1] = ratio_value
        except:
            print(ratio_name)
    return ratio_data

def elem_product_list(data):
    ratio_data = pd.DataFrame()
    columns = data.columns
    first_list, secound_list = elem_combinations_list(columns)
    for elem1, elem2 in zip(first_list, secound_list):
        ratio_name_1 = elem1 + "*" + elem2
        ratio_name_2 = elem2 + "*" + elem1
        ratio_name = elem1 + "/" + elem2
        try:
            first_data = data[elem1]
            secound_data = data[elem2]
            ratio_value = first_data*secound_data
            ratio_data[ratio_name_1] = ratio_value
        except:
            print(ratio_name)
    return ratio_data
#################################################計算関連

def feature_making_metamorphic(feature_setting, train_x, sc, pca, ICA):
        ###########################################特徴量の設定
        #X_allに特徴量をまとめていく
        X_all_train = pd.DataFrame()

        if feature_setting.setting_X_raw == "on":
            X_all_train = pd.concat([X_all_train, train_x], axis = 1)
        elif feature_setting.setting_X_log == "on":
            #LOG変換
            train_x_log = train_x.apply(lambda x:np.log10(x))
            X_all_train = pd.concat([X_all_train, train_x_log], axis = 1)

        #Product log変換するかもTest
        if feature_setting.setting_X_product == "on":
            if feature_setting.setting_X_product_log == "on":
                train_x_product = elem_product_list(train_x).apply(lambda x:np.log10(x))
            else:
                train_x_product = elem_product_list(train_x)
            X_all_train = pd.concat([X_all_train, train_x_product], axis = 1)

        #Ratio log変換するかもTest
        if feature_setting.setting_X_ratio == "on":
            if feature_setting.setting_X_ratio_log == "on":
                train_x_ratio = elem_ratio_list(train_x).apply(lambda x:np.log10(x))
            else:
                train_x_ratio = elem_ratio_list(train_x)
            X_all_train = pd.concat([X_all_train, train_x_ratio], axis = 1)

        #Standard_scaler&次元削減の初期設定
        #PCA, ICAの変数を決める。Xのlen(変数)で決める
        n_components=len(train_x.columns)
        ############################################## PCA・ICAを一元素でやっても意味がないので， 1の時はやらない
        if n_components != 1:
            #PCA, ICAの名前の準備
            PCA_index = []
            ICA_index = []
            for num in range(n_components):
                PCA_index.append("PC"+ str(num + 1))
                ICA_index.append("IC"+ str(num + 1))

            #Standard_scaler&次元削減 #indexを指定するのをわすれない
            #LOG変換されたもの or Not
            if feature_setting.setting_NORMAL_OR_LOG == "on":
                s_scaler_train = pd.DataFrame(sc.transform(train_x_log), index = train_x.index)
            else:
                s_scaler_train = pd.DataFrame(sc.transform(train_x), index = train_x.index)

            #PCA
            if feature_setting.setting_PCA == "on":
                pca_train = pd.DataFrame(pca.fit_transform(s_scaler_train), index = train_x.index, columns=PCA_index)
                #Concat
                X_all_train = pd.concat([X_all_train, pca_train], axis = 1)
            #ICA
            if feature_setting.setting_ICA == "on":
                ICA_train = pd.DataFrame(ICA.fit_transform(s_scaler_train), index = train_x.index, columns=ICA_index)
                #Concat
                X_all_train = pd.concat([X_all_train, ICA_train], axis = 1)
        else :
            sc = 0
            pca = 0
            ICA = 0
        ############################################## PCA・ICAを一元素でやっても意味がないので，　1の時はやらない
        #再度代入設定
        train_x = X_all_train.copy()
        return train_x

def predict_cv_metamorphic(Models, feature_setting, Fold_num, train_x):

    models = Models.model
    sc_s = Models.sc
    pca_s = Models.pca
    ICA_s = Models.ICA

    #predict append
    train_predict_append = []
    train_predict_dist_append = []
    #cross_validation
    for num in range(Fold_num):
        model_now = models[num]
        sc_now = sc_s[num]
        pca_now = pca_s[num]
        ICA_now = ICA_s[num]

        ###########################################特徴量の設定
        train_x_make_feature = feature_making_metamorphic(feature_setting, train_x, sc_now, pca_now, ICA_now)
        ###########################################特徴量の設定

        #predict
        predicted_train_y = model_now.predict(train_x_make_feature)
        if feature_setting.ML_algorithm_name == 'NGBoost':
            predicted_train_y_dist = model_now.pred_dist(train_x_make_feature).params['scale']
        else:
            index_number = train_x_make_feature.shape[0]
            predicted_train_y_dist = np.zeros([1, index_number]).flatten()

        #predict append
        train_predict_append.append(predicted_train_y)
        train_predict_dist_append.append(predicted_train_y_dist)

    #テストデータに対する予測値の平均をとる
    pred_train = np.mean(train_predict_append, axis = 0)
    train_predict_dist_append = pd.DataFrame(train_predict_dist_append)**2 # ２乗
    train_predict_dist_append = train_predict_dist_append.mean()
    pred_test_dist = train_predict_dist_append**(1/2)
    feature_name = train_x_make_feature.columns

    return pred_train, pred_test_dist, feature_name

#################################################モデルの定義

def predict_model(elem, X_use, y,  path_all_share, path_figure_all, path_all_models, mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist):

    #データの代入
    X_elem = X_use
    #目的関数はLOG空間上で評価したいので，初めからLOG
    y_elem = y[elem]

    print("###############################")
    print(elem)

    ##################################################model_featureのsettingをread
    #pickleからModel_settingの呼び出し
    try:
        pkl_opne_path_1 = path_all_models + "/Model_setting.pkl"
        feature_setting = pickle.load(open(pkl_opne_path_1 , mode='rb'))
    except:
        st.write(elem)
        st.write("/Model_setting.pkl maybe not exist")

    #pickleからModel_settingの呼び出し
    try:
        Fold_num = 4
        models = []
        sc_s = []
        pca_s = []
        ICA_s = []
        for num in range(Fold_num):
            #モデルのread
            pkl_opne_path_1 = path_all_models + "/Models/"+str(num)+".pkl"
            Models_fold = pickle.load(open(pkl_opne_path_1 , mode='rb'))

            models.append(Models_fold.model)
            sc_s.append(Models_fold.sc)
            pca_s.append(Models_fold.pca)
            ICA_s.append(Models_fold.ICA)

    except:
        print("/Predict_model.pkl maybe not exist")
    Models_predict = Models(models, sc_s, pca_s, ICA_s)

    #####################################CrossVaridationによるアンサンブルモデル作成，Predict，Score
    Fold_num = 4
    pred_y_log, pred_y_dist_log, feature_name = predict_cv_metamorphic(Models_predict, feature_setting, Fold_num, X_use)
    pred_y = pd.Series(pred_y_log, index = y_elem.index).apply(lambda x: 10**x)
    pred_y_dist_log = pd.Series(pred_y_dist_log.values, index = y_elem.index)

    ########################################関数の定義
    pred_data = pd.DataFrame()
    mobile_amount_data = pd.DataFrame()

    #######################testデータ
    #この際のtest_yを代入していく

    pred_data = pd.concat([y_elem, X_use], axis = 1)
    pred_data["RAW"] = y_elem
    pred_data["predict"] = pred_y
    pred_data["predict_dist"] = pred_y_dist_log
    ##########PMの値をかけて，orderをppmに直して行く
    #normalizeのためのデータをread
    for_normalize_data = pd.read_excel("../List/Primitive_Mantle _ C1 Chondrite.xlsx", index_col=0)
    for_normalize_data = for_normalize_data.drop(columns=['Unnamed: 1']).T
    #一部のデータはErrorになるので，先にdrop
    for_normalize_data = for_normalize_data.drop(['F', 'In', 'Cl', 'Ge'])
    try:
        PM_value = for_normalize_data[elem].values
        pred_data["Mobile_amount"] = (y_elem - pred_y)*PM_value
    except:
        print(elem + " is not normalized by PM")
        pred_data["Mobile_amount"] = y_elem - pred_y


    sample_name_now = pred_data["Mobile_amount"].index

    ######mobile_amount
    mobile_amount_data[elem] = pred_data["Mobile_amount"].copy()

    #出力とspidergram
    mobile_data_compile[elem] = pred_data["Mobile_amount"]
    spidergram_data_compile[elem] = pred_data["predict"]

    if feature_setting.ML_algorithm_name == 'NGBoost':
        mobile_data_compile_dist[elem] = pred_data["predict_dist"]
        spidergram_data_compile_dist[elem] = pred_data["predict_dist"]
    else:
        mobile_data_compile_dist[elem] = pred_data["predict_dist"] ## model scoreにできるように変換
        spidergram_data_compile_dist[elem] = pred_data["predict_dist"] ## model scoreにできるように変換

    return pred_data, mobile_amount_data, mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist


def model_main(mobile_elem, immobile_elem, Raw_metamorphic_rock, Raw_metamorphic_rock_location, model_folder_name, mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist):
    ########################################################元素の数を計算
    #不動元素
    #一回全て代入
    use_element = mobile_elem + immobile_elem

    ########################################################サンプル数を計算
    index_names = Raw_metamorphic_rock[use_element].dropna().index

    Sample_info = Raw_metamorphic_rock_location.T[index_names].T
    Sample_info_num = Sample_info.groupby('SAMPLE_INFO').size()
    Sample_info_num_sum = Sample_info.groupby('SAMPLE_INFO').size().sum()
    Sample_DataBase_num = Sample_info.groupby('DataBase').size()

    ########################################################サンプル数を計算
    #Check point
    Data_info = pd.Series()

    print("####################")
    print("Element number : " + str(len(use_element)))
    print("Mobile elem : " + str(mobile_elem))
    print("Mobile elem : " + str(len(mobile_elem)))
    print("Immobile elem : " + str(immobile_elem))
    print("Immobile elem : " + str(len(immobile_elem)))
    print("####################")
    print("Sample_Tectonic_Setting : ")
    print(Sample_info_num)
    print("Sample_num : " + str(Sample_info_num_sum))
    print("Sample_DataBase : ")
    print(Sample_DataBase_num)
    print("####################")

    Data_info["Element number"] = len(use_element)
    Data_info["Mobile elem"] = mobile_elem
    Data_info["Mobile elem num"] =  len(mobile_elem)
    Data_info["Immobile elem : "] = immobile_elem
    Data_info["Immobile elem num:"] = len(immobile_elem)
    Data_info = pd.concat([Data_info, Sample_info_num])
    Data_info["SUM num "] = Sample_info_num_sum
    Data_info = pd.concat([Data_info, Sample_DataBase_num])

    #######################################################ディレクトリの作成
    #pathとフォルダの準備
    path_1 = model_folder_name + "/" + str(immobile_elem).strip("[").strip("]").strip("'")
    path_2 = "/" + str(mobile_elem).strip("[").strip("]").strip("'")
    #全体のpathの設定
    path_all_share = path_1 + path_2

    #全体のpathの設定
    path_all_models = path_1 + path_2
    print(path_all_models)

    #図を保存するディレクトリの作成
    path_4 = "/Figure"
    path_figure_all = path_all_share + path_4
    make_dirs(path_figure_all)


    #######################################################使うデータの整理
    #目的yと入力データXを設定
    data___ = Raw_metamorphic_rock[use_element].dropna()
    #X_use
    X_use = data___[immobile_elem]
    #y 目的変数
    y = data___[mobile_elem]

    #今回の流体移動元素の定義
    elem = mobile_elem[0]
    ##########################################################################
    ##########################################################################
    ##########################################################################モデルによる推定
    ########################################MODEL
    pred_data, mobile_amount_data, mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist = predict_model(elem, X_use, y,  path_all_share, path_figure_all, path_all_models, mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist)
    ########################################MODEL  fin

    return mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist

@st.cache
def predict_protolith(mobile_elem_list, immobile_elem, Raw_metamorphic_rock, Raw_metamorphic_rock_location, now_model_folder_name):

    # DEFINE 箱を準備
    # 物質移動量の計算
    mobile_data_compile = pd.DataFrame(index = Raw_metamorphic_rock.index, columns = mobile_elem_list)
    mobile_data_compile_dist = pd.DataFrame(index = Raw_metamorphic_rock.index, columns = mobile_elem_list)
    # 推定した値を入れる
    spidergram_data_compile = pd.DataFrame()
    spidergram_data_compile_dist = pd.DataFrame()
    for mobile_elem in mobile_elem_list:
        if mobile_elem not in immobile_elem:
            mobile_elem = [mobile_elem]
            mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist = model_main(mobile_elem, immobile_elem, Raw_metamorphic_rock, Raw_metamorphic_rock_location, now_model_folder_name, mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist)

    spidergram_data_compile[immobile_elem]=Raw_metamorphic_rock[immobile_elem].copy()

    return mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist
