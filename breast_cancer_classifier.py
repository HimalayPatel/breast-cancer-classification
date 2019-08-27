import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get data and load it into Pandas dataframe
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer_df=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))

# Apply visualization techniques on the data to better see the data
sns.pairplot(cancer_df,hue='target',vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
plt.show()
sns.countplot(cancer_df['target'])
plt.show()
sns.scatterplot(x='mean area',y='mean perimeter',hue='target',data=cancer_df)
plt.show()
sns.heatmap(cancer_df.corr(),annot=True)
plt.show()

# Train the model (splitting into 70-30) and using SVM
X=cancer_df.drop(['target'],axis=1)
y=cancer_df['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
cancer_svc=SVC()
cancer_svc.fit(X_train,y_train)
y_predict=cancer_svc.predict(X_test)
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)
plt.show()

# Improvement #1 applying Normalization (Feature Scaling- Unity-based normalization)
min_train=X_train.min()
range_train=(X_train-min_train).max()
X_train_scaled=(X_train-min_train)/range_train
sns.scatterplot(x=X_train['mean area'],y=X_train['mean smoothness'],hue=y_train)
plt.show()
sns.scatterplot(x=X_train_scaled['mean area'],y=X_train_scaled['mean smoothness'],hue=y_train)
plt.show()
min_test=X_test.min()
max_test=X_test.max()
range_test=(max_test-min_test)
X_test_scaled=(X_test-min_test)/range_test
cancer_svc.fit(X_train_scaled,y_train)
y_predict=cancer_svc.predict(X_test_scaled)
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)
plt.show()
print(classification_report(y_test,y_predict))

# Improvement #2- Tuning C and gamma hyperparameters using GridSearchCV
param_grid={'C':[0.1, 1, 10, 10],'gamma':[1, 0.1, 0.01, 0.001], 'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid_predict=grid.predict(X_test_scaled)
cm=confusion_matrix(y_test,grid_predict)
sns.heatmap(cm,annot=True)
plt.show()
print(classification_report(y_test,grid_predict))
