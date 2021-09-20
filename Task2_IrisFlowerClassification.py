#!/usr/bin/env python
# coding: utf-8

# #                    TASK-2  Iris_Flowers_Classification_ML_Project 

# Author: Paritosh Raikar

# ## Import libraries

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
sns.set()


# ## Load the Dataset

# In[39]:


Iris_data = pd.read_csv("Iris.csv")
Iris_data


# ## Data Preprocessing

# In[40]:


Iris_data.info()


# Data Analysis

# In[41]:


Iris_data.describe()


# In[42]:


Iris_data.isnull().sum()


# Checking NUll values in the Dataset

# In[43]:


Iris_data.count()


# Summary of the Dataset

# In[45]:


Iris_data['Species'].value_counts()


# ## Data Visualization

# In[46]:


fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(data=Iris_data,width=0.5, ax=ax, fliersize=3)


# In[47]:


sns.pairplot(data=Iris_data, hue="Species")
plt.show()


# ## Finding Co-relation

# In[48]:


Iris_data.corr()


# In[49]:


plt.figure(figsize=(8,7))
sns.heatmap(Iris_data.corr(),annot=True)
plt.show()


# ## Correlation

# In[50]:


Iris_data['SepalLengthCm'].hist()
Iris_data['SepalWidthCm'].hist()
Iris_data['PetalLengthCm'].hist()
Iris_data['PetalWidthCm'].hist()


# In[51]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.barplot(x='Species', y='SepalLengthCm', data=Iris_data)
plt.subplot(2,2,2)
sns.barplot(x='Species', y='PetalLengthCm', data=Iris_data)
plt.subplot(2,2,3)
sns.barplot(x='Species', y='SepalWidthCm', data=Iris_data)
plt.subplot(2,2,4)
sns.barplot(x='Species', y='PetalWidthCm', data=Iris_data)


# In[52]:


setosa=Iris_data[Iris_data['Species']=="Iris-setosa"]
versicolor=Iris_data[Iris_data['Species']=="Iris-versicolor"]
virginica=Iris_data[Iris_data['Species']=="Iris-virginica"]

plt.figure(figsize=(10,10))
plt.scatter(setosa['PetalLengthCm'],setosa['PetalWidthCm'], c="blue", label="Iris-setosa", marker='*')
plt.scatter(versicolor['PetalLengthCm'],versicolor['PetalWidthCm'], c="red", label="versicolor", marker='^')
plt.scatter(virginica['PetalLengthCm'],virginica['PetalWidthCm'], c="green", label="virginica", marker='<')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Lenght vs Petal Width", fontsize=15)
plt.legend()
plt.show()


# In[53]:


setosa=Iris_data[Iris_data['Species']=="Iris-setosa"]
versicolor=Iris_data[Iris_data['Species']=="Iris-versicolor"]
virginica=Iris_data[Iris_data['Species']=="Iris-virginica"]

plt.figure(figsize=(10,10))
plt.scatter(setosa['SepalLengthCm'],setosa['SepalWidthCm'], c="blue", label="Iris-setosa", marker='*')
plt.scatter(versicolor['SepalLengthCm'],versicolor['SepalWidthCm'], c="red", label="versicolor", marker='^')
plt.scatter(virginica['SepalLengthCm'],virginica['SepalWidthCm'], c="green", label="virginica", marker='<')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Lengt vs Sepal Width", fontsize=15)
plt.legend()
plt.show()


# In[54]:


x=Iris_data.drop(columns="Species")
y=Iris_data["Species"]


# In[56]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4, random_state = 1)


# In[57]:


x_train.head()


# In[58]:


x_test.head()


# In[60]:


y_train.head()


# In[61]:


y_test.head()


# In[62]:


print("x_train ", len(x_train))
print("x_test ", len(x_test))
print("y_train ", len(y_train))
print("y_test ", len(y_test))


# ## Model Building 

# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[74]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[64]:


LogisticRegression(solver='lbfgs', max_iter=100)


# In[76]:


predict = model.predict(x_test)
print('predicted the vlaues on the test data',predict)


# In[77]:


y_test_pred=model.predict(x_test)
y_train_pred=model.predict(x_train)


# In[78]:


print("Training Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Test Accuracy    :",  accuracy_score(y_test, y_test_pred))


# In[86]:


from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors = 9, metric='minkowski', p = 2 )
Classifier.fit(x_train,y_train)


# ## DecisionTreeClassifier

# In[88]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

X, y = load_iris(return_X_y=True)

clf = DecisionTreeClassifier(max_depth = 5)

clf.fit(X, y)

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(clf,
           feature_names = fn, 
           class_names=cn,
           filled = True);


# ## Confusion Matrix

# In[90]:


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)


titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


#  #                                        THANK YOU....!!!  
