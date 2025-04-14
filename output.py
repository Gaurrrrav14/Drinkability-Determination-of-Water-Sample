import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

main_df = pd.read_csv("water_potability.csv")
df = main_df.copy() #Making a acopy of the dataset to work on

df.head()

df.shape

df.columns

df.dtypes

df.info()

df.describe()

#Distribution of Potability Values
df['Potability'].value_counts(normalize=True)

print(df.nunique())

#Finding the missing value on different columns
print(df.isnull().sum())

ax = sns.countplot(x = "Potability",data= df, saturation=0.8)
plt.xticks(ticks=[0, 1], labels = ["Not Potable", "Potable"])
plt.show()

print(df["Potability"].value_counts())
df["Potability"].value_counts().plot(kind="pie", autopct='%1.1f%%',figsize=(8,8));

#Distribution of Different Parameters
df.drop('Potability', axis=1).hist(figsize=(12,8));

df.corr()

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), cmap='Blues', annot=True)
plt.show()

index = 0
plt.figure(figsize=(20, 20))
for feature in df.columns:
    if feature != "Potability":
        index += 1
        plt.subplot(4, 3, index)
        sns.boxplot(x='Potability', y=feature, data=df)

sns.pairplot(df, hue="Potability")

df[df['Potability']==1].describe()

df[df['Potability']==0].describe()

msno.bar(df, figsize = (16,5),color = '#FD7702')
plt.show()

df[df['Sulfate'].isnull()]
df[df['ph'].isnull()]
df[df['Trihalomethanes'].isnull()]

#Filling the missing values with their column mean gerouped by different Portability Status

df['ph']=df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate']=df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes']=df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))

df.isna().sum()

fig = msno.bar(df, color='#0474BA')

# Models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Evaluation & CV Libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score


X = df.drop('Potability', axis=1)
y = df['Potability']

#Partitioning the dataset in training set and test set
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Standardizing and Normalising the data set
scale = StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

finalResults = []
#Performing Logistic Regression
LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = LR_model.predict(X_test)
y_train_pred = LR_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("LR",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=LR_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

#Performing Support Vector Machine
SVC_model = SVC()
SVC_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = SVC_model.predict(X_test)
y_train_pred = SVC_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("SVC",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=SVC_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SVC_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

#Performing K-Neighbors Classifier
KNN_model = KNeighborsClassifier(n_neighbors=10)
KNN_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = KNN_model.predict(X_test)
y_train_pred = KNN_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("KNN",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=KNN_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=KNN_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

#Performing Decision Tree Classifier
DTC_model = DecisionTreeClassifier()
DTC_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = DTC_model.predict(X_test)
y_train_pred = DTC_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("DTC",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=DTC_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DTC_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

#Performing Naive Bayes
GNB_model = GaussianNB()
GNB_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = GNB_model.predict(X_test)
y_train_pred = GNB_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("GNB",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=GNB_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GNB_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

#Performing Random Forest Classifier
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = RF_model.predict(X_test)
y_train_pred = RF_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("RF",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=RF_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

#Performing Gradient Boosting Classifier
XGB_model = GradientBoostingClassifier()
XGB_model.fit(X_train, y_train)

#Predicting on test dataset
y_pred = XGB_model.predict(X_test)
y_train_pred = XGB_model.predict(X_train)

log1_acc = accuracy_score(y_test, y_pred)
finalResults.append(("XGB",log1_acc))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=XGB_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=XGB_model.classes_)
disp.plot(cmap=plt.cm.Blues)
print(log1_acc)

finalResults.sort(key=lambda k:k[1],reverse=True)
finalResults

#Plotting the accuracy score against the models
final = pd.DataFrame(finalResults)
ax= sns.barplot(data=final,x=final[1], y=final[0], orient='h', palette='Blues', saturation=0.4, linewidth=0.5)
ax.set(xlabel='Accuracy Score', ylabel='Models')
plt.show()



