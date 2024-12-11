# GUI Application for Protolith Reconstruction Models (PRMs)
[![DOI](https://zenodo.org/badge/721911387.svg)](https://zenodo.org/doi/10.5281/zenodo.10183973)

https://github.com/MSrakugo/PRM_Streamlit/assets/51268793/51af29c7-9d2a-4c77-bd2e-91dbab76971d

**Full Movie**: https://github.com/MSrakugo/PRM_Streamlit/blob/main/DEMO/PRM_Demo.mp4

**Web application**: https://protolith-reconstruction-models-gui-app.streamlit.app/

## Overview
This project has developed a **Graphical User Interface (GUI) application for Protolith Reconstruction Models (PRMs)** using Python and Streamlit. PRMs utilize machine learning algorithms to estimate the original protolith composition from the composition of altered/metamorphosed rocks (the product), allowing for the **quantitative estimation of element transfer**.

This application not only allows users to **easily handle PRMs** through a user-friendly GUI but also offers **tools for the normalization and visualization of compositional data** through simple file drop functionality.

このプロジェクトでは、Streamlitを用いて**原岩組成復元モデル（PRMs）用のグラフィカルユーザーインターフェース（GUI）アプリケーション**を開発しました。PRMsは、機械学習アルゴリズムを利用して、個々の試料に対して、原岩（出発物質）の組成を元素移動を被った変質/変成岩（生成物質）の組成から推定し、**元素移動を定量的に推定** することが可能です。

このアプリケーションは、**ユーザーフレンドリーなGUI** を通じてPRMsを簡単に扱うことができるだけでなく、**組成データの正規化と可視化のためのツール** も提供します。これらのツールは、シンプルな**ファイルドロップ機能**を通じて利用可能です。

## UPDATE History

Current status: Almost update have already finished (2024/09/18).

2024/12/11 Correction to Bag: R^2 factor miscalculation (Matsuno+ 2022 values is correct), PM normalized Ratio data (e.g., "Zr_") incorrection.
2024/09/15 ver1.3 **Major Update**: Model construction based on GUI are enable

2023/11/23 The first release of PRMs

## Paper
This PRM methodology is based on the research paper:
* Matsuno, S., Uno, M., Okamoto, A. et al. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022). [https://doi.org/10.1038/s41598-022-05109-x](https://doi.org/10.1038/s41598-022-05109-x)

* Specifically, elements in the altered/metamorphosed rock composition that have not undergone element transfer (for example, Th, Nb) are considered as traces of the original protolith (the starting material). These elements are input into a machine learning model constructed using a dataset of original protolith compositions. The estimated protolith composition obtained through the machine learning model, compared with the known altered/metamorphosed rock composition, allows for the calculation of quantitative element transfer.
* 具体的には、変質/変成岩組成の中で元素移動を被らなかった元素群（例えばTh, Nbなど）を、原岩（出発物質）の組成の痕跡と見なし、原岩組成データセットによって構築された機械学習モデルに入力します。機械学習モデルから得られた原岩組成の推定値を、既知である変質/変成岩組成と比較することで、定量的な元素移動量が求められます。

Key words:

**Geochemistry, Fluid-rock interaction, Element transfer, Quantitative evaluation, Machine-learning techniques**

## Application Examples

Seafloor Alteration Governed by Oceanic Crustal Age and Redox Conditions: Insights from Machine Learning-based Elemental Transfer Analyses. Matsuno, S., Uno, M., Okamoto, A [(Under review)](https://essopenarchive.org/users/708320/articles/692946-seafloor-alteration-governed-by-oceanic-crustal-age-and-redox-conditions-insights-from-machine-learning-based-elemental-transfer-analyses)

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

1. **Download the ZIP file or Clone the Repository**
    - To download as ZIP, visit [GitHub repository](https://github.com/MSrakugo/PRM_Streamlit), click on 'Code' and then 'Download ZIP'. After downloading, extract the ZIP file.
      - Alternatively, clone the repository using:
        ```bash
        git clone https://github.com/MSrakugo/PRM_Streamlit
        ```
    - Navigate to the folder using Terminal:
      ```bash
      cd [Your Repository Directory]
      ```

2. **(Optional) Build a Virtual Environment in Python**
    - It's recommended to create a virtual environment to keep dependencies required by different projects separate and organized. Use the following commands to create and activate a virtual environment:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      ```

3. **Install Dependencies**
    - Install the necessary libraries specified in the `requirements.txt` file:
      ```bash
      pip install -r requirements.txt
      ```

4. **Run the Application**
    - Launch the application with the following command:
      ```bash
      streamlit run PRM_App_Main.py
      ```

    - The application will start running locally on your machine. Open your web browser and navigate to the local URL provided (usually `http://localhost:8501`).


## 使い方（日本語版は簡略版です）
1. **ZIPファイルのダウンロード**
    - ファイルを展開した後、**ターミナルを使用してフォルダに移動します**：
      ```bash
      cd [Your Repository Directory]
      ```

2. **（Optional）Pythonで仮想環境を構築**
    - 省略。Pythonを普段から使用している方は仮想環境の構築を推奨します。

3. **依存関係のインストール**
    - 必要なライブラリをインストールするために、以下のコマンドを実行します：
      ```bash
      pip install -r requirements.txt
      ```

4. **アプリケーションの実行**
    - アプリケーションを起動するには、次のコマンドを実行します：
      ```bash
      streamlit run PRM_App_Main.py
      ```

もし上記の内容で分からない点があれば、お気軽にX(Twitter) @mtsn_stsh またはメールでお問い合わせください。

## Author

Satoshi Matsuno, Tohoku university, Japan

松野哲士 東北大学環境科学研究科

HP: https://sites.google.com/view/matsunos/home


## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
