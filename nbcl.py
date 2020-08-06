import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('linkedin.csv')
print(df)

x = df.drop(columns=['Tech/ Non-tech'])
y = df['Tech/ Non-tech']
print(x)
print(y)


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()  
x= x.apply(label_encoder.fit_transform)
print(x)
y= label_encoder.fit_transform(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))

z=label_encoder.inverse_transform(y_pred)
print(z)
