# Product Category Prediction from Listing Metadata

This project aims to build a robust multi-class classification model that predicts the **bottom-level product category** of listings on an e-commerce platform based on their metadata. The work was carried out as part of a college-industry collaboration, using real-world product listing data provided by a company partner. The dataset was shared under academic usage terms and all sensitive identifiers (e.g., product IDs) have been anonymized.

---

## 📦 Dataset Overview

The dataset consists of product metadata extracted from an e-commerce platform. Each row in the dataset represents a unique product listing. The primary objective is to predict the `bottom_category_text` for each product using various product features.

### Key Columns:
- `title`: Product title (text)
- `description`: Product description (text)
- `tags`: List of associated tags (text)
- `type`, `room`, `craft_type`, `recipient`, `material`, `occasion`, etc.: Structured categorical fields
- `primary_color_text`, `secondary_color_text`: Color features
- `bottom_category_text`: Target variable (multi-class)

The dataset contains thousands of unique listings, and the labels are hierarchically structured, such as:
```
clothing.girls_clothing.skirts ⟶ clothing
```

---

## 🧠 Project Objectives

1. **Perform Exploratory Data Analysis (EDA)** to identify class imbalance, data quality, and informative features.
2. **Preprocess text fields** (e.g., cleaning, tokenization, lemmatization, TF-IDF vectorization).
3. **Encode categorical features** using One-Hot or Label Encoding.
4. **Build classification models** using Logistic Regression, Support Vector Machines (SVM), and Random Forest.
5. **Evaluate model performance** using accuracy, classification report, and visualizations.

---

## 🛠️ Workflow Summary

1. **Mount Google Drive (for Colab users)** to access data.
2. **Import libraries** from Pandas, NumPy, Matplotlib, NLTK, and Scikit-learn.
3. **Perform preprocessing**:
   - Text normalization (stopwords removal, lemmatization)
   - Feature transformation (TF-IDF for text, encoding for categorical features)
4. **Model building**:
   - Train/test split
   - Train Logistic Regression, SVM, and Random Forest classifiers
   - Evaluate each model on accuracy and confusion matrices
5. **Model interpretation**:
   - Chi-squared feature selection
   - Classification report and feature importance (if applicable)

---

## 🧪 Model Performance

Each classifier was evaluated using standard metrics:

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

The best-performing model was selected based on validation performance. Additional experimentation was performed with hyperparameter tuning and feature selection.

---

## 📁 Repository Structure

```
product-category-prediction/
│
├── data/
│   ├── raw/              # Anonymized source data
│   └── processed/        # Cleaned and transformed datasets
│
├── notebooks/
│   └── category_prediction_final.ipynb  # Main notebook
│
├── models/
│   └── model.pkl         # (Optional) Serialized trained model
│
├── README.md
└── requirements.txt
```

---

## 📌 Important Notes

- Product identifiers and personally identifiable data have been removed or masked.
- The data was provided by a company partner for academic and educational use only. Redistribution is not permitted without consent.
- All code is documented and reproducible in a standard Python environment.

---

## 📚 Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Key packages:
- pandas, numpy, matplotlib, seaborn
- nltk, scikit-learn
- scipy

---

## 🤝 Acknowledgment

Special thanks to our company partner for providing access to this dataset for academic purposes. This project was completed as part of coursework in applied data science and machine learning.
