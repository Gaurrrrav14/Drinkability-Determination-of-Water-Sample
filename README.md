# Drinkability-Determination-of-Water-Sample
💧 Drinkability Determination
This project focuses on predicting the potability (drinkability) of water using various physicochemical properties. By analyzing features such as pH, hardness, solids, and contaminants, we use machine learning algorithms to determine whether water is safe for human consumption.

📊 Dataset
Source: water_potability.csv

Features:

pH

Hardness

Solids

Chloramines

Sulfate

Conductivity

Organic Carbon

Trihalomethanes

Turbidity

Potability (Target): 0 = Not drinkable, 1 = Drinkable

Missing Values:

Present in ph, Sulfate, and Trihalomethanes, filled with class-wise mean values.

🔍 Project Workflow
1. 🧼 Data Preprocessing
Handled missing values using class-mean imputation

Standardized features using StandardScaler

Visualized distributions using histograms, boxplots, heatmaps, and pie charts

2. 🧠 Models Trained
Algorithm	Description
Logistic Regression	Baseline linear classifier
Support Vector Machine	Non-linear decision boundaries
K-Nearest Neighbors	Distance-based classifier (K=10)
Decision Tree	Rule-based tree structure
Naive Bayes	Probabilistic approach
Random Forest	Ensemble of decision trees
Gradient Boosting	Advanced boosting-based ensemble
All models were evaluated on accuracy and visualized using confusion matrices and classification reports.

3. 📈 Model Comparison
Models were ranked by test accuracy and compared using bar plots.

🛠️ Technologies & Libraries
Python 3.9+

pandas, numpy, seaborn, matplotlib, missingno

scikit-learn

tensorflow (basic import, not heavily used here)

Jupyter Notebook / .py script compatible

📊 Example Visualizations
Correlation heatmap of features

Potability distribution

Boxplots by potability

Confusion matrices for model performance

✅ Results
Best-performing models: Random Forest and Gradient Boosting

Accuracy scores:

Random Forest: ~83%

Gradient Boosting: ~81%

Others: 70–80%


🧠 Future Enhancements
Integrate deep learning models (DNN)

Build a Streamlit or Flask web dashboard

Use real-time IoT water sensors as input

Feature selection or PCA for dimensionality reduction

