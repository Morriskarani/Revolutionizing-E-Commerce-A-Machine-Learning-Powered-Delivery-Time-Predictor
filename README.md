Revolutionizing E-Commerce: A Machine Learning Powered Delivery Time Predictor


📌 Project Overview

This project focuses on predicting whether an e-commerce product will be delivered on time or late using machine learning. By analyzing customer behavior, logistics details (such as warehouse location, shipment mode, and weight), and order history, the system helps optimize delivery logistics for better customer satisfaction. The project follows the CRISP-DM methodology and achieves a prediction accuracy of 69.45% using a Decision Tree model on a Kaggle dataset with 10,999 records. The solution includes a web application deployed on Railway for real-time predictions.

🧠 Machine Learning Models Used





Decision Tree ✅ (Best model – Accuracy: 69.45%)



Random Forest



Logistic Regression



K-Nearest Neighbors (KNN)



Multi-Layer Perceptron (MLP)



XGBoost



Ensemble Voting Classifier

The Decision Tree model was selected after comparing performance metrics (accuracy, precision, recall) using GridSearchCV for hyperparameter tuning.

🛠 Technologies & Tools





Languages & Libraries: Python (pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn)



Web Framework: Flask (for the web application interface)



Version Control: Git & GitHub



Development Environment: Jupyter Notebook



Deployment: Railway



Data Visualization: Matplotlib, Seaborn

📊 Features





SMOTE for Class Balancing: Addresses imbalanced data to improve model performance.



GridSearchCV for Model Tuning: Optimizes hyperparameters for the best accuracy.



Evaluation Metrics: Includes ROC Curve, Confusion Matrix, and detailed evaluation reports.



Dashboard UI: Real-time predictions via a Flask-based web app (deployed on Railway).



Feature Importance Analysis: Identifies key predictors like Discount Offered and Weight in Grams.



CRISP-DM Methodology: Structured approach from Business Understanding to Deployment.

📁 Dataset

The project utilizes a Kaggle dataset containing 10,999 e-commerce shipment records. Key features include:





Customer ID



Warehouse Block



Mode of Shipment



Customer Care Calls



Customer Rating



Cost of Product



Prior Purchases



Product Importance



Gender



Discount Offered



Weight in Grams



Reached.on.Time_Y.N (Target Variable)

The dataset is preprocessed using SMOTE to handle class imbalance and is available for download from Kaggle E-Commerce Shipping Data (replace with the exact link if different).
