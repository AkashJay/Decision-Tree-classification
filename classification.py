import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataset =  pd.read_csv('result.csv')
x1 = dataset.iloc[:, :-1].values
y1 = dataset.iloc[:, 9].values



#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableEncoder_x = LabelEncoder()
x1[:, 1] = lableEncoder_x.fit_transform(x1[:, 1])
x1[:, 2] = lableEncoder_x.fit_transform(x1[:, 2])
x1[:, 3] = lableEncoder_x.fit_transform(x1[:, 3])
x1[:, 4] = lableEncoder_x.fit_transform(x1[:, 4])
x1[:, 5] = lableEncoder_x.fit_transform(x1[:, 5])

lableEncoder_y = LabelEncoder()

df_x1 = pd.DataFrame(x1)
y1 = lableEncoder_y.fit_transform(y1)


onehotencoder = OneHotEncoder(categorical_features= [1])
x1 = onehotencoder.fit_transform(x1).toarray()

onehotencoder = OneHotEncoder(categorical_features= [4])
x1 = onehotencoder.fit_transform(x1).toarray()

onehotencoder = OneHotEncoder(categorical_features= [11])
x1 = onehotencoder.fit_transform(x1).toarray()

onehotencoder = OneHotEncoder(categorical_features= [17])
x1 = onehotencoder.fit_transform(x1).toarray()

onehotencoder = OneHotEncoder(categorical_features= [23])
x1 = onehotencoder.fit_transform(x1).toarray()

df_x = pd.DataFrame(x1)

#spliting data into traning and testing test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)

df_y_test = pd.DataFrame(y_test)

#decision tree classification
from sklearn.tree import DecisionTreeClassifier
classifer = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifer.fit(x_train, y_train)

#predicting the test set result
y_pred = classifer.predict(x_test)

df_y_pred = pd.DataFrame(y_pred)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
aa = accuracy_score(y_test, y_pred)





