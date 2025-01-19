# Predicting the Success of Bank Telemarketing Campaigns

## Project Overview
This project aims to predict whether a client will subscribe to a bank term deposit based on telemarketing campaign data. The dataset includes a variety of features such as demographic details, financial information, and campaign-related metrics. By analyzing this data and building predictive models, we can gain valuable insights to optimize marketing strategies and improve overall success rates.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Analyzed individual variables to identify trends and patterns using visualizations such as histograms, bar charts, and boxplots.
- **Bivariate Analysis**: Explored relationships between variables and the target variable using correlation heatmaps, cross-tabulations, and scatter plots.

### 2. Feature Engineering
- Preprocessed the data by encoding categorical variables, handling missing values, and scaling numerical features.
- Created new features and transformations to enhance predictive power.

### 3. Pipelines and Column Transformers
- Developed **pipelines** to automate preprocessing steps and ensure a streamlined workflow.
- Used **ColumnTransformer** to handle different data types efficiently.

### 4. Feature Selection
- Applied **Recursive Feature Elimination (RFE)** to select the most important features for the model, reducing complexity and improving efficiency.

### 5. Model Building
Trained multiple machine learning models, including:
- Logistic Regression
- SGDClassifier
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- XGBClassifier

### 6. Model Evaluation
Evaluated model performance using the following metrics:
- Accuracy Score
- Precision, Recall, and F1 Score
- ROC-AUC
- Confusion Matrix

The **XGBoost Classifier** achieved the best performance with an **F1 score of 75.5**, making it the model of choice for this project.

### 7. Hyperparameter Tuning
- Optimized the XGBoost model using **RandomizedSearchCV**, tuning parameters such as learning rate, number of estimators, and maximum depth to enhance performance.

### 8. Final Predictions
- Used the tuned XGBoost model to generate predictions, which were submitted for evaluation.

---

## Results
- **Best Model**: XGBoost Classifier
- **F1 Score**: 75.5
- **Grade**: S

The project demonstrates how predictive modeling can support marketing campaigns by identifying potential clients likely to subscribe to a term deposit.

---

## Key Learnings
1. Importance of thorough EDA to uncover trends and relationships in the data.
2. Effectiveness of pipelines and column transformers in automating preprocessing tasks.
3. The impact of feature selection and hyperparameter tuning on model performance.
4. How to evaluate models comprehensively using multiple metrics.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

---

## Conclusion
This project highlights the power of machine learning in optimizing business processes. The insights derived from the analysis can help banks allocate resources more effectively and achieve better marketing outcomes.

---

## Future Work
- Explore additional feature engineering techniques to further improve model performance.
- Test other advanced machine learning models and ensemble techniques.
- Deploy the model as a web application for real-time predictions.

---

### Contact
For any queries or collaborations, feel free to reach out!
