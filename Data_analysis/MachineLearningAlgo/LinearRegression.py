# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# importing train test split from scikit-learn for training purpose
from sklearn.model_selection import train_test_split
# importing linear-regression from scikit-learn for appling linear regression
from sklearn.linear_model import LinearRegression
# for accuracy checking
from sklearn import metrics

# importing dataset and extracting the dependent(Y) and independent(X) variables
salary_data = pd.read_csv("E:/Machine_learning/Data_analysis/MachineLearningAlgo/Salary_Data.csv")
X = salary_data.iloc[:, :-1:].values
Y = salary_data.iloc[:, 1].values

# sns.distplot(salary_data['YearsExperience'],kde=False,bins=10)
# sns.countplot(y='YearsExperience', data=salary_data)
sns.barplot(x='YearsExperience', y='Salary', data=salary_data)
plt.show()
sns.heatmap(salary_data.corr())
plt.show()

# splitting data set into train set and test set
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=1/3,random_state=0)

# fitting training set into simple linear regression
lr = LinearRegression()
lr.fit(X_train,Y_train)

# pridicting the test result
y_pred = lr.predict(X_test)
# print(y_pred)

# using matplotlib for visualizing the test results
plt.scatter(X_train, Y_train, color = 'blue')
plt.plot(X_train, lr.predict(X_train), color = 'red')
plt.title('Salary ~ Experience (Train set)')
plt.xlabel('Tears of Experience')
plt.ylabel('Salary')
# plt.show()

# finding the residuals and computing the accuracy
print('MAE:', metrics.mean_absolute_error(Y_test, y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, y_pred))
print('RMAE:', np.sqrt(metrics.mean_absolute_error(Y_test, y_pred)))