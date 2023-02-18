# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
# importing train test split from scikit-learn for training purpose
from sklearn.model_selection import train_test_split
# standered scaler for scaling the input
from sklearn.preprocessing import StandardScaler
# importing linear-regression from scikit-learn for appling Logestic regression
from sklearn.linear_model import LogisticRegression
# for accuracy checking
from sklearn.metrics import confusion_matrix

# importing dataset and extracting the dependent(Y) and independent(X) variables
salary_data = pd.read_csv("E:/Machine_learning/Data_analysis/MachineLearningAlgo/SocialNetworkAds.csv")
X = salary_data.iloc[:,[2,3]].values
Y = salary_data.iloc[:, 4].values


# sns.heatmap(salary_data.corr())
# plt.show()

# splitting data set into train set and test set
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.25,random_state=0)

# feature Scaling (If we have too extreme values that can distrub our pridiction)
scx = StandardScaler()
X_train = scx.fit_transform(X_train)
X_test = scx.transform(X_test)

# fitting training set into simple Logestic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)

# pridicting the test results
y_pred = lr.predict(X_test)

# visualising the training set 
x_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:,1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logestic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()

# visualising the test set results
x_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:,1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logestic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()

# confusion matrix evaluation
cm = confusion_matrix(Y_test,y_pred)
print(cm)