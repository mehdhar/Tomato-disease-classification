# Tomato Leaf Disease Classification Using Transfer Learning and Feature Concatenation  
Official Implementation of the IET Image Processing Paper (2022)

---

## Overview

This repository provides the implementation of:

"Tomato leaf disease classification by exploiting transfer learning and feature concatenation" 
Published in *IET Image Processing, 2022*  
DOI: https://doi.org/10.1049/ipr2.12397  

The proposed method extracts deep features from MobileNetV2 and NASNetMobile, concatenates them, reduces dimensionality using Kernel PCA, and classifies images using traditional machine learning classifiers (SVM, RF, MLR).  

This hybrid DL–ML pipeline is designed to perform accurately even on small datasets, outperforming deeper CNNs when training data is limited.

---

## Proposed Method

The method consists of four major stages:

1. Deep Feature Extraction  
   - Lightweight CNN models (MobileNetV2, NASNetMobile)  
   - Extracted feature vectors are flattened  

2. Feature Concatenation
   - Combine features from both CNNs into a single hybrid vector  

3. Dimensionality Reduction
   - Kernel PCA (Linear kernel performed best)

4. Classification  
   - Evaluated classifiers:  
     - Support Vector Machine (SVM)  
     - Random Forest (RF)  
     - Multinomial Logistic Regression (MLR)  
   - MLR achieved the best overall performance  

---

## Dataset

This work uses a subset of the PlantVillage dataset:  
https://github.com/spMohanty/PlantVillage-Dataset

Number of classes: **6**  
- Bacterial Spot  
- Late Blight  
- Leaf Mold  
- Septoria Spot  
- Yellow Leaf Curl Virus  
- Healthy  

Images resized to **224×224** to match CNN input requirements.

---

## Repository Structure

```
Tomato-Disease-Classification/
├── extract_features.py        # MobileNetV2 / NASNetMobile feature extraction
├── concatenate_features.py    # Feature fusion module
├── reduce_dimension.py        # Kernel PCA implementation
├── train_classifier.py        # SVM, Random Forest, MLR training
├── evaluate.py                # Evaluation and metrics
├── requirements.txt           # Dependencies
│
├── data/
│   └── dataset/               # PlantVillage subset (user must add manually)
│
└── README.md
```

---

## Installation

```
pip install -r requirements.txt
```

---

## How to Run

### 1. Extract deep features
```
python extract_features.py
```

### 2. Concatenate features
```
python concatenate_features.py
```

### 3. Apply Kernel PCA
```
python reduce_dimension.py
```

### 4. Train classifier
```
python train_classifier.py
```

---

## Results

Performance on Tomato Leaf Dataset:

| Classifier | Accuracy |
|-----------|----------|
| RF        | 88–96%   |
| SVM       | 89–94%   |
| **MLR**   | **97%**  |

Confusion matrices and ROC curves are provided in the paper (pages 10–12).  
:contentReference[oaicite:1]{index=1}

---

## Citation

If you use this code, please cite the original paper:

```
Al-Gaashani, M.S.A.M., Shang, F., Muthanna, M.S.A., Khayyat, M., Abd El-Latif, A.A.
Tomato leaf disease classification by exploiting transfer learning and feature concatenation.
IET Image Processing, 2022.
DOI: 10.1049/ipr2.12397
```

---

## Notes

- The dataset is **not included** due to licensing.  
- Code is intended for **research and academic use**.  
- This repository reproduces the methodology described in the publication.

