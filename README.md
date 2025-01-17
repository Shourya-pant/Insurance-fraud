# 🛡️ Health Insurance Fraud Detection Using Machine Learning  

Fraud detection in the health insurance sector is critical to reducing financial losses and improving service delivery. This project aims to detect fraudulent claims using machine learning algorithms and data preprocessing techniques, achieving a robust solution for healthcare fraud prevention.  

---

## 🚀 Project Overview  
- **Goal:** Predict potentially fraudulent healthcare providers based on submitted claims.  
- **Approach:** Analyze healthcare data, identify key fraud-indicating features, and develop classification models for fraud detection.  
- **Key Outcome:** The Support Vector Machine (SVM) model emerged as the best algorithm with an **accuracy of 91.7%**.  

---

## 📊 Dataset Description  
The project used healthcare claims data comprising the following:  
1. **Inpatient Claims:** Details of patients admitted to hospitals, including admission/discharge dates and diagnoses.  
2. **Outpatient Claims:** Records of patients visiting hospitals without admission.  
3. **Beneficiary Details:** KYC details such as health conditions and regions of residence.  

### Data Preprocessing Steps:  
- Handled missing values and noisy data.  
- Encoded categorical variables using one-hot encoding.  
- Scaled numerical variables for consistency.  
- Addressed data imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.  

---

## 🧠 Machine Learning Approach  
### Algorithms Tested:  
- Random Forest  
- Support Vector Machine (SVM)  
- Naïve Bayes  
- Decision Tree  
- Gradient Boosting  

### Best Model:  
The **SVM algorithm** was the most effective, achieving:  
- **Accuracy:** 91.7%  
- **Precision:** 96%  
- **Recall:** 94%  
- **F1-Score:** 95%  

Other models, such as Random Forest and Gradient Boosting, also performed well, with accuracies exceeding or close to 90%.  

---

## 📈 Visualizations  
- Correlation heatmaps for feature relationships.  
- Confusion matrices to evaluate model predictions.  
- Precision-Recall and ROC curves for model performance.  

---

## 🛠️ Technologies Used  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SMOTE  

---

## 🧪 How to Run  


### Steps to Execute  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/health-insurance-fraud-detection.git  
   cd health-insurance-fraud-detection  
   ```  
2.Train the model

3.Run file appy.py

---

## 🎯 Results  
This project demonstrates the effectiveness of machine learning models in detecting fraudulent health insurance claims, with the SVM model leading in performance metrics.  

---

## 🤝 Contributing  
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.  

---

## ✨ Acknowledgements  
- Dataset sourced from [Kaggle].  
- Inspired by the need to enhance fraud detection mechanisms in healthcare systems globally.
