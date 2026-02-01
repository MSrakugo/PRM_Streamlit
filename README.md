# GUI Application for Protolith Reconstruction Models (PRMs)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15792787.svg)](https://doi.org/10.5281/zenodo.15792787)


https://github.com/MSrakugo/PRM_Streamlit/assets/51268793/51af29c7-9d2a-4c77-bd2e-91dbab76971d

**Full Movie**: https://github.com/MSrakugo/PRM_Streamlit/blob/main/DEMO/PRM_Demo.mp4

**Web application**: https://protolith-reconstruction-models-gui-app.streamlit.app/

## Overview
This project has developed a **Graphical User Interface (GUI) application for Protolith Reconstruction Models (PRMs)** using Python and Streamlit. PRMs utilize machine learning algorithms to estimate the original protolith composition from the composition of altered/metamorphosed rocks (the product), allowing for the **quantitative estimation of element transfer**.

This application not only allows users to **easily handle PRMs** through a user-friendly GUI but also offers **tools for the normalization and visualization of compositional data** through simple file drop functionality.

このプロジェクトでは、Streamlitを用いて**原岩組成復元モデル（PRMs）用のグラフィカルユーザーインターフェース（GUI）アプリケーション**を開発しました。PRMsは、機械学習アルゴリズムを利用して、個々の試料に対して、原岩（出発物質）の組成を元素移動を被った変質/変成岩（生成物質）の組成から推定し、**元素移動を定量的に推定** することが可能です。

このアプリケーションは、**ユーザーフレンドリーなGUI** を通じてPRMsを簡単に扱うことができるだけでなく、**組成データの正規化と可視化のためのツール** も提供します。これらのツールは、シンプルな**ファイルドロップ機能**を通じて利用可能です。

## UPDATE History

Current status: Almost update have already finished (2026/02/01).

2026/02/01 ver2.1 **Minor Update**: Debag error in major element setting.


2025/07/03 ver2.0 **Major Update**: 
major update including an environment update (constructed with Poetry) and [open notebooks for XAI](/0_NoteBook/)

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

* [Matsuno, Satoshi, Masaoki Uno, and Atsushi Okamoto. 2025. “The Control of Oceanic Crustal Age and Redox Conditions on Seafloor Alteration: Examples from a Quantitative Comparison of Elemental Mass Transport in the South and Northwest Pacific.” Chemical Geology 678 (122651): 122651.](https://doi.org/10.1016/j.chemgeo.2025.122651)
    * [Source code](https://github.com/MSrakugo/Matsuno_2025a_Seafloor_Alteration)

* [Matsuno, Satoshi, Masaoki Uno, and Atsushi Okamoto. 2025. “Low‐dimensional Controls on Oceanic Basalt Geochemistry Revealed by Regression‐based Machine Learning Models.” Journal of Geophysical Research: Machine Learning and Computation 2 (4).](https://doi.org/10.1029/2025jh000700.)
    * This study used [this notebook](https://github.com/MSrakugo/PRM_Streamlit/blob/main/0_NoteBook/5_SHAP_Explaination_250604.ipynb) for model interpretation. You can run this notebook after model construction.

## Features
* **Protolith Reconstruction Models (PRMs)**: Accurately estimate the protolith composition from a limited number of input elements of altered or metamorphosed samples. The default model focuses on basalt trace-element compositions.
  * These default models are developed using LightGBM algorithms and are trained with datasets encompassing various types of basalts, including mid-ocean ridge basalts, ocean-island basalts, and volcanic arc basalts.
  * In the near future, I plan to update this application to enable model construction with your customized datasets.
* **User-Friendly Interface**: The application is intuitively designed, ensuring ease of use for individuals with different levels of technical expertise.

## Requirements

* **Python 3.11**
* **Poetry 2.0 or higher**:
  [Official documentation](https://python-poetry.org/docs/)

## Getting Started (with Poetry)

These instructions assume you have met the requirements listed above.

1.  **Clone the Repository**
    * Clone the repository to your local machine:
        ```bash
        git clone https://github.com/MSrakugo/PRM_Streamlit
        ```
    * Navigate to the newly created directory:
        ```bash
        cd PRM_Streamlit
        ```

2.  **Install Dependencies**
    * Poetry will read the `pyproject.toml` file, create a dedicated virtual environment using Python 3.11, and install all necessary libraries automatically.
        ```bash
        poetry install --no-root
        ```

3.  **Run the Application**
    * To run the Streamlit application within the virtual environment managed by Poetry, use the `poetry run` command:
        ```bash
        poetry run streamlit run PRM_App_Main.py
        ```
    * The application will start, and you can access it in your web browser, usually at `http://localhost:8501`.

### (Optional) Working directly within the virtual environment

Instead of prefixing every command with `poetry run`, you can activate the virtual environment. This method activates the environment within your current shell session.

* First, get the command to activate the environment:
    ```bash
    poetry env activate
    ```
* This will print an activation command. Copy and paste it to run it. It will look something like this:
    ```bash
    source /path/to/your/virtualenv/bin/activate
    ```
* Once activated, you can run commands directly. To deactivate and return to your normal shell, run:
    ```bash
    deactivate
    ```
4.  **Construct and Use Protolith Reconstruction Models**
    * You can construct PRMs and predict protolith composition from alterd/metamorphosed rocks by yourself.
    * This GUI contain below contents:
        1. To construct PRMs by your suitable data
        2. Evaluation and Explain selected PRMs with test dataset
        3. Read your compositional data and apply PRMs
        4. Preprocessing & Visualization with primitive mantle normalization

---

## 動作環境

* **Python 3.11**
* **Poetry 2.0 以上**

## 使い方 (Poetry版)

上記の動作環境が準備されていることを前提としています。

1.  **リポジトリのクローン**
    * `git`コマンドでリポジトリをローカルにコピーします。
        ```bash
        git clone https://github.com/MSrakugo/PRM_Streamlit
        ```
    * ターミナルで、作成されたフォルダに移動します。
        ```bash
        cd PRM_Streamlit
        ```

2.  **依存関係のインストール**
    * Poetryが`pyproject.toml`ファイルを元に、Python 3.11を使用したプロジェクト専用の仮想環境を自動で作成し、必要なライブラリをすべてインストールします。
        ```bash
        poetry install --no-root
        ```

3.  **アプリケーションの実行**
    * Poetryが管理する仮想環境内でアプリケーションを起動するには、`poetry run`コマンドを使います。
        ```bash
        poetry run streamlit run PRM_App_Main.py
        ```
    * アプリケーションが起動し、WebブラウザでローカルURL（通常は `http://localhost:8501`）にアクセスできるようになります。

### （任意）仮想環境内で直接作業する

毎回 `poetry run` を先頭につける代わりに、仮想環境を有効化（activate）して作業することができます。現在使っているシェルのセッション内で、環境を有効化する方法です。

* まず、有効化のためのコマンドを表示させます。
    ```bash
    poetry env activate
    ```
* 実行すると、有効化用のコマンドが出力されます。それをコピー＆ペーストして実行してください。以下のような形式です。
    ```bash
    source /path/to/your/virtualenv/bin/activate
    ```
* 有効化されたら、直接コマンドを実行できます。環境を無効化して元の状態に戻すには、以下のコマンドを実行します。
    ```bash
    deactivate
    ```

---
もし上記の内容で分からない点があれば、お気軽にX(Twitter) @mtsn_stsh またはメールでお問い合わせください。

## Author

Satoshi Matsuno, Tohoku university, Japan

松野哲士 東北大学環境科学研究科

HP: https://sites.google.com/view/matsunos/home


## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
