#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: PRM_Application_Main
Begin: Tue Mar  1 23:06:08 2022
Final update: 2023/11/21

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: .
"""

import streamlit as st

def page_footer():
    st.write("---")
    st.subheader("Cite articles")
    st.markdown("1. Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022). https://doi.org/10.1038/s41598-022-05109-x")
    st.markdown("2. Matsuno, S. Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models. (2023). ")
    st.write("---")
    st.caption("Press release in Japanese: https://www.tohoku.ac.jp/japanese/2022/02/press20220210-01-machine.html")
    st.caption("Auther: Satoshi Matsuno (Tohoku univ., Japan)")
    st.caption("If you find any errors, please point them out to us through Email (satoshi.matsuno.p2@dc.tohoku.ac.jp) or Github (https://github.com/MSrakugo/PRM_application)")
