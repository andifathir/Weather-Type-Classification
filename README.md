<h1 align="center">🌦️ WEATHER TYPE CLASSIFICATION</h1>

<div align="center">
  <img src="https://github.com/andifathir/Weather-Type-Classification/blob/master/images/Weathers.png?raw=true" alt="Weather Banner" width="520">
  <p>
    <small>
      Image Source: <a href="https://www.pbslearningmedia.org/resource/buac17-k2-sci-ess-diffweather/different-types-of-weather/
      ">Access Here</a>
    </small>
  </p>
</div>

---

<h1 align="center">📑 TABLE OF CONTENT</h1>

1. [Project Overview](#-project-overview-)
   - [Background](#background)
   - [Project Objectives](#project-objectives)
2. [Dataset Source](#-dataset-source-)
3. [Features Description](#-features-description-)
4. [Preprocessing & Modeling](#-preprocessing--modeling-)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architectures](#model-architectures)
5. [Installation Guide](#-installation-guide-)
   - [Main Software](#main-software)
   - [Dependencies](#dependencies)
   - [Running the Streamlit App](#running-the-streamlit-app)
6. [Results & Evaluation](#-results--evaluation-)
7. [Streamlit Prediction System](#-streamlit-prediction-system-)
   - [Interface Overview](#interface-overview)
   - [Live Demo](#live-demo)
8. [Author](#-author-)

---

<h1 align="center">🌍 Project Overview 🌍</h1>

This project focuses on **Weather Type Classification** using **tabular meteorological data**.  
The system compares several **deep learning architectures specialized for tabular data**, aiming to identify the most effective model for predicting weather conditions.

The models implemented in this project include:

- **Multi-Layer Perceptron (MLP)**
- **TabNet**
- **FT-Transformer**

All models are deployed into an **interactive Streamlit dashboard** that allows users to experiment with different inputs and observe prediction confidence.

---

### Background

Weather classification plays a critical role in:

- Climate monitoring  
- Environmental analysis  
- Early weather warning systems  
- Decision support in agriculture and transportation  

However, tabular weather data often contains **outliers, mixed feature types, and non-linear relationships**, making it a strong candidate for modern deep learning approaches such as **attention-based** and **transformer-based** models.

---

### Project Objectives

1. **Build a multi-class weather classification system**
2. **Compare traditional and modern deep learning models** for tabular data
3. **Visualize prediction probabilities** for interpretability
4. **Deploy an interactive Streamlit application** for real-time inference

---

<h1 align="center">📊 Dataset Source 📊</h1>

The dataset used in this project is obtained from Kaggle:

- **Dataset Name**: *Weather Type Classification*
- **Link**: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification
- **Data Type**: Tabular
- **Task**: Multi-class classification

The dataset contains both **numerical and categorical features** and includes intentionally **extreme and outlier values**, making preprocessing and robust modeling essential.

---

<h1 align="center">🧾 Features Description 🧾</h1>

| Feature Name | Type | Description |
|-------------|------|-------------|
| Temperature | Numeric | Temperature in °C, including extreme values |
| Humidity | Numeric | Humidity percentage, may exceed 100% |
| Wind Speed | Numeric | Wind speed (km/h), includes extreme values |
| Precipitation (%) | Numeric | Probability/intensity of precipitation |
| Atmospheric Pressure | Numeric | Pressure in hPa |
| UV Index | Numeric | Ultraviolet radiation intensity |
| Visibility (km) | Numeric | Visibility distance |
| Cloud Cover | Categorical | Cloud coverage description |
| Season | Categorical | Season when data was recorded |
| Location | Categorical | Type of location |
| **Weather Type** | Categorical | **Target variable** |

---

<h1 align="center">⚙️ Preprocessing & Modeling ⚙️</h1>

### Data Preprocessing

- Numerical features are **scaled** to improve model convergence
- Categorical features are **encoded** using:
  - Label Encoding (TabNet, FT-Transformer)
  - ColumnTransformer (MLP)
- Dataset is split into **training and testing sets**
- Outliers are intentionally preserved to test model robustness

---

### Model Architectures

#### 1. Multi-Layer Perceptron (MLP)
- Dense neural network
- Uses `ColumnTransformer`
- Baseline deep learning model for tabular data

#### 2. TabNet
- Attention-based architecture
- Performs **feature selection during training**
- Highly interpretable

#### 3. FT-Transformer
- Transformer adapted for tabular data
- Separate embeddings for numerical & categorical features
- Strong performance on structured datasets

---

<h1 align="center">🔧 Installation Guide 🔧</h1>

### Main Software
- Python **3.10+**
- VSCode / Google Colab (for training)
- Streamlit (for deployment)

---

### Installation (PDM)

This project uses **PDM** for dependency management.

```bash
pdm install
```

### Running the Streamlit App

```bash
pdm run streamlit run app.py
```

Once running, open the provided local URL in your browser.

# 📈 Results & Evaluation 📈

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Class probability distribution

### Overall Metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|------|----------|-----------------|--------------|----------------|
| MLP | 0.91 | 0.91 | 0.91 | 0.91 |
| TabNet | 0.90 | 0.91 | 0.90 | 0.90 |
| FT-Transformer | 0.90 | 0.91 | 0.90 | 0.90 |

---

### Class-wise F1-Score Comparison

| Weather Type | MLP | TabNet | FT-Transformer |
|-------------|-----|--------|----------------|
| Cloudy | 0.89 | 0.88 | 0.88 |
| Rainy | 0.90 | 0.90 | 0.89 |
| Snowy | 0.93 | 0.93 | 0.93 |
| Sunny | 0.92 | 0.91 | 0.92 |

---

### Observations

- MLP achieves the highest overall accuracy (91%), serving as a strong baseline.
- TabNet and FT-Transformer show comparable performance with attention-based mechanisms.
- All models perform particularly well on the Snowy class.
- Performance differences are marginal, indicating the dataset is well-structured for tabular learning.

## 📉 Training Curves

### Loss Curves per Model

| MLP | TabNet | FT-Transformer |
|-----|--------|----------------|
| ![MLP Loss](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/mlp_loss.png?raw=true) | ![TabNet Loss](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/tabnet_loss.png?raw=true) | ![FT-Transformer Loss](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/FT_Transformer_Loss.png?raw=true) |

---

### Accuracy Curves per Model

| MLP | TabNet | FT-Transformer |
|-----|--------|----------------|
| ![MLP Accuracy](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/mlp_acc.png?raw=true) | ![TabNet Accuracy](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/tabnet_acc.png?raw=true) | ![FT-Transformer Accuracy](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/FT_Transformer_Acc.png?raw=true) |

---

## 🧮 Confusion Matrix Comparison

| MLP | TabNet | FT-Transformer |
|-----|--------|----------------|
| ![MLP Confusion Matrix](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/mlp_cm.png?raw=true) | ![TabNet Confusion Matrix](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/tabnet_cm.png?raw=true) | ![FT-Transformer Confusion Matrix](https://github.com/andifathir/Weather-Type-Classification/blob/master/images/FT_Transformer_CM.png?raw=true) |

---


# 🖥️ Streamlit Prediction System 🖥️

The Streamlit dashboard allows users to:
- Select the prediction model (MLP / TabNet / FT-Transformer)
- Input numerical & categorical weather features
- View:
  - Predicted weather type
  - Confidence score
  - Probability distribution across all classes

## Interface Overview

<div align="center">
  <img src="https://github.com/andifathir/Weather-Type-Classification/blob/master/images/interface1.png?raw=true" width="600"><br>
  <img src="https://github.com/andifathir/Weather-Type-Classification/blob/master/images/interface2.png?raw=true" width="600"><br>
  <img src="https://github.com/andifathir/Weather-Type-Classification/blob/master/images/interface3.png?raw=true" width="600">
</div>


# 👤 Author 👤
Andi Fathir Muzakki Diningrat (202210370311278) 
🎓 Informatics Engineering  
📍 Indonesia  
🔗 [GitHub](https://github.com/andifathir)