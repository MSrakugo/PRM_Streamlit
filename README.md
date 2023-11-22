# GUI Application for Protolith Reconstruction Models (PRMs)

## Overview
This project has developed a **Graphical User Interface (GUI) application for Protolith Reconstruction Models (PRMs)** using Python and Streamlit. PRMs utilize machine learning algorithms to estimate the original protolith composition from the composition of altered/metamorphosed rocks (the product), allowing for the **quantitative estimation of element transfer**.

This application not only allows users to **easily handle PRMs** through a user-friendly GUI but also offers **tools for the normalization and visualization of compositional data** through simple file drop functionality.

このプロジェクトでは、PythonとStreamlitを用いて**原岩組成復元モデル（PRMs）用のグラフィカルユーザーインターフェース（GUI）アプリケーション**を開発しました。PRMsは、機械学習アルゴリズムを利用して、個々の試料に対して、原岩（出発物質）の組成を元素移動を被った変質/変成岩（生成物質）の組成から推定し、**元素移動を定量的に推定** することが可能です。

このアプリケーションは、**ユーザーフレンドリーなGUI** を通じてPRMsを簡単に扱うことができるだけでなく、**組成データの正規化と可視化のためのツール** も提供します。これらのツールは、シンプルな**ファイルドロップ機能**を通じて利用可能です。

## Paper
This PRM methodology is based on the research paper:
* Matsuno, S., Uno, M., Okamoto, A. et al. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022). [https://doi.org/10.1038/s41598-022-05109-x](https://doi.org/10.1038/s41598-022-05109-x)

* Specifically, elements in the altered/metamorphosed rock composition that have not undergone element transfer (for example, Th, Nb) are considered as traces of the original protolith (the starting material). These elements are input into a machine learning model constructed using a dataset of original protolith compositions. The estimated protolith composition obtained through the machine learning model, compared with the known altered/metamorphosed rock composition, allows for the calculation of quantitative element transfer.
* 具体的には、変質/変成岩組成の中で元素移動を被らなかった元素群（例えばTh, Nbなど）を、原岩（出発物質）の組成の痕跡と見なし、原岩組成データセットによって構築された機械学習モデルに入力します。機械学習モデルから得られた原岩組成の推定値を、既知である変質/変成岩組成と比較することで、定量的な元素移動量が求められます。

Key words:

**Geochemistry, Fluid-rock interaction, Element transfer, Quantitative evaluation, Machine-learning techniques**

## Features
* **Protolith Reconstruction Models (PRMs)**: Accurately estimate the protolith composition from a limited number of input elements of altered or metamorphosed samples. The default model focuses on basalt trace-element compositions.
  * These default models are developed using LightGBM algorithms and are trained with datasets encompassing various types of basalts, including mid-ocean ridge basalts, ocean-island basalts, and volcanic arc basalts.
  * In the near future, I plan to update this application to enable model construction with your customized datasets.
* **User-Friendly Interface**: The application is intuitively designed, ensuring ease of use for individuals with different levels of technical expertise.

## Requirement
Before running the application, ensure you have the following requirements installed:

- Python 3.x
- Streamlit
- LightGBM
- Other dependencies listed in `requirements.txt`

Note: Python 3.x and pip should be installed on your system.

## How to use

To use this application, follow these steps:

1. **Clone the Repository（ファイルの複製）**
```bash
git clone https://github.com/MSrakugo/PRM_Streamlit
```
* **You can also download the ZIP file. (ZIPファイルのダウンロードでも可)**

After downloading, use Terminal to navigate to the folder（ダウンロード後、Terminalでフォルダまで移動）
```bash
cd [Your Repository Directory]
```

2. **Install Dependencies**
pip install -r requirements.txt





## Author

Satoshi Matsuno (松野哲士)

Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
