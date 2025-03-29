# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation
2. Splitting Data
3. Model Training
4. Model Evaluation and Visualization

## Program:
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Niranjan S
RegisterNumber:  24900209
*/
```

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()


df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())


x=df.drop('target',axis=1)
y= df['target']


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

sgd_clf = SGDClassifier(max_iter = 1000, tol= 1e-3)
sgd_clf.fit(x_train, y_train)#train the classifier on the training data

y_pred = sgd_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)# evaluate the classifier's accuracy
print(f"Accuracy:{accuracy:.3f}")

#calculate the confusion matrix
cf=confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cf)
```
## Output:
![prediction of iris species using SGD Classifier](sam.png)
![370345555-91435a3b-4ba7-4dda-92da-afd66210769b](https://github.com/user-attachments/assets/27e4648e-cd58-4408-87ec-36d0c1d65d8e)

![368834566-e86c22c4-2827-4454-893e-1833fb288993](https://github.com/user-attachments/assets/bcea5ce7-bb50-443b-9f59-4629ec76a85a)

![368834596-a05c5ed4-a05b-4266-8b69-d11e191b6150](https://github.com/user-attachments/assets/9150b30e-4e59-40ed-9ab8-cf11ed922f6a)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
