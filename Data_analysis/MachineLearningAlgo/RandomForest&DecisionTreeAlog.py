# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# importing train test split from scikit-learn for training purpose
from sklearn.model_selection import train_test_split
# importing Decision Tree and Random Forest from scikit-learn for appling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# for accuracy checking
from sklearn.metrics import classification_report,confusion_matrix


# importing dataset and extracting the dependent(Y) and independent(X) variables
kyphosis = pd.read_csv("E:/Machine_learning/Data_analysis/MachineLearningAlgo/kyphosis.csv")
x = kyphosis.iloc[:,[1,2,3]]
y = kyphosis['Kyphosis']

# Data analysis
sns.barplot(x = 'Kyphosis', y = 'Age', data = kyphosis)
# plt.show()
sns.pairplot(kyphosis, hue= 'Kyphosis',palette='Set1')
# plt.show()

# Visuslising the Dataset
plt.figure(figsize=(25,7))
sns.countplot(x='Age', hue='Kyphosis', data=kyphosis, palette='Set1')
plt.show()

# splitting data set into train set and test set
X_train, X_test, Y_train, Y_test =train_test_split(x, y, test_size=0.3,random_state=100)
# print(x.head)

# Training decision tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train,Y_train)

# Pridicting the Modle of decision tree
pridiction = dtree.predict(X_test)
# print(pridiction)

# Evaulation the modle
print(classification_report(Y_test,pridiction))
print(confusion_matrix(Y_test,pridiction))

# Training Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)

rfc_pred = rfc.predict(X_test)

# Evaulation the modle(Random forest)
print(confusion_matrix(Y_test, rfc_pred))
print(classification_report(Y_test, rfc_pred))