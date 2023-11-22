#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: PRM_Application_Main
Begin: Tue Mar  1 23:06:08 2022
Final update:

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Citation:
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
