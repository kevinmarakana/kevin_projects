data set link 

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Credit Card Fraud Detection

This project demonstrates a complete workflow for detecting credit card fraud using machine learning techniques. The dataset used is a publicly available credit card fraud detection dataset, which includes both legitimate and fraudulent transactions.

## Dataset Description
- **File Name**: `creditcard.csv`
- **Columns**: 31 columns
  - `Time`: Transaction time in seconds (dropped during preprocessing).
  - `V1` to `V28`: Principal components obtained via PCA.
  - `Amount`: Transaction amount (standardized during preprocessing).
  - `Class`: Target column where `0` indicates normal transactions and `1` indicates fraudulent transactions.

## Workflow

### 1. Data Loading and Exploration
- Load the dataset using Pandas.
- Display the first and last few rows to understand the structure.
- Summarize the dataset with:
  - Number of rows and columns.
  - Information about data types and null values.

### 2. Preprocessing
- **Standardization**:
  - Standardized the `Amount` column using `StandardScaler`.
- **Feature Dropping**:
  - Removed the `Time` column as it was not significant for model training.
- **Duplicate Removal**:
  - Checked for and removed duplicate rows to ensure clean data.

### 3. Class Imbalance Handling
Due to the highly imbalanced nature of the dataset, both **undersampling** and **oversampling** techniques were applied:

#### Undersampling
- Downsampled the majority class (normal transactions) to balance it with the minority class (fraudulent transactions).
- Re-trained and evaluated multiple classifiers.

#### Oversampling
- Used Synthetic Minority Oversampling Technique (SMOTE) to generate synthetic samples for the minority class.
- Re-trained and evaluated multiple classifiers.

### 4. Classification Models
Trained and evaluated the following machine learning models:
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Random Forest Classifier
- Naive Bayes
- k-Nearest Neighbors (KNN)

#### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 5. Model Saving and Deployment
- Trained a Decision Tree Classifier on the oversampled data.
- Saved the trained model using `joblib`.
- Loaded the saved model for prediction on new data.

### 6. Example Prediction
- Provided an example transaction vector to the saved model.
- Output whether the transaction is **Normal** or **Fraudulent**.

## Dependencies
The project uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`
- `joblib`

## Usage
1. Ensure that the `creditcard.csv` dataset is in the same directory as the script.
2. Run the script to:
   - Preprocess the data.
   - Train and evaluate multiple classifiers.
   - Save the best-performing model.
3. Use the saved model for predictions with a new transaction vector.

## Results
- The classifiers were evaluated on their ability to handle imbalanced data.
- The use of SMOTE improved the performance metrics for fraudulent transaction detection.

## Future Improvements
- Implement additional techniques for handling imbalanced data (e.g., ensemble methods).
- Optimize hyperparameters of classifiers for better accuracy.
- Explore deep learning models for fraud detection.

## License
This project is for educational purposes and follows the dataset's usage terms.

---
Feel free to reach out for further clarifications or enhancements!

