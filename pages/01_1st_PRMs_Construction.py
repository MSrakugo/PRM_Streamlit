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

import streamlit as st
import pandas as pd
import numpy as np
import openpyxl

import plotly.express as px
import matplotlib.pyplot as plt

import PRM_liblary as prm
import PRM_Predict_liblary as prm_predict
import Library_preprocessing as preprocessing
#import Library_model_construction as construction_PRM
import PRM_App_Library as App_Library


#################################################################################### sidebar
st.header("1st step: PRMs Construction (Optional)")

st.write("Coming soon...")

#################################################################################### Main
App_Library.page_footer()
