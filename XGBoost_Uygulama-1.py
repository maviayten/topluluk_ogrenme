#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


file_path = "C:\\Users\\User\\Downloads\\diabetes_binary_health_indicators_BRFSS2021.csv"
diabetes_data = pd.read_csv(file_path)


diabetes_data.head()


print("Veri setinin boyutları:", diabetes_data.shape)

print("\nSütunlardaki eksik değer sayıları:")
missing_values = diabetes_data.isnull().sum()
print(missing_values)

print("\nSütunların veri tipleri:")
print(diabetes_data.dtypes)


# In[3]:


diabetes_data.head()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

print("Temel İstatistikler:\n", diabetes_data.describe())




# In[5]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


X = diabetes_data.drop('Diabetes_binary', axis=1)
y = diabetes_data['Diabetes_binary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


xgb_clf = XGBClassifier(random_state=42)


xgb_clf.fit(X_train, y_train)


y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Doğruluk (Accuracy):", accuracy)
print("Sınıflandırma Raporu:\n", report)


# In[6]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X = diabetes_data.drop('Diabetes_binary', axis=1)
y = diabetes_data['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

xgb_clf = XGBClassifier(random_state=42)

xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Doğruluk (Accuracy):", accuracy)
print("Sınıflandırma Raporu:\n", report)


# In[7]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb_clf = XGBClassifier()


param_grid_xgb = {
    'n_estimators': [100, 200, 300],         
    'learning_rate': [0.01, 0.1, 0.2],     
    'max_depth': [3, 4, 5],                  
    'min_child_weight': [1, 2, 3],           
    'subsample': [0.5, 0.7, 1.0],         
    'colsample_bytree': [0.5, 0.7, 1.0]      
}

grid_search_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=3, scoring='accuracy', verbose=1)
grid_search_xgb.fit(X_train, y_train)


best_parameters_xgb = grid_search_xgb.best_params_
best_score_xgb = grid_search_xgb.best_score_

print("En İyi Parametreler:", best_parameters_xgb)
print("En İyi Skor:", best_score_xgb)


# In[8]:


from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


best_xgb_clf = grid_search_xgb.best_estimator_
best_xgb_clf.fit(X_train, y_train)


y_pred_xgb = best_xgb_clf.predict(X_test)


print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred_xgb))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_xgb))


# In[9]:


y_pred_proba = best_xgb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='GBM (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




