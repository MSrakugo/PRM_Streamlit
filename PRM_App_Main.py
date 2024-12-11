#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2023/11/25

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""

import streamlit as st
import PRM_App_Library as App_Library


#################################################################################### Main
st.header("Protolith Reconstruction Models (PRMs)")
st.markdown("This is a **GUI interface** for Protolith Reconstruction Models.")
st.markdown("You can **construct PRMs** and **predict protolith composition** from alterd/metamorphosed rocks by yourself")

st.markdown('> **1st step**   : (Optional) To **construct** PRMs by your suitable data')
st.markdown('> **2nd step**   : (Optional) **Evaluation and Explain** selected PRMs with test dataset')
st.markdown('> **3rd step**   : **Read** your compositional data and **Run** PRMs')

# PRMsの解説

#################################################################################### Main

# footer
App_Library.page_footer()
