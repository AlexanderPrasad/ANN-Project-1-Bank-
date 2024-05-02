#Alexander Prasad
#Georgia Institute Of Technoogy
#Four Layer Artificial Neural Network Implementation
#November 19th, 2023

import numpy as np
import pandas as pd
import tensorflow as tf


dataset = pd.read_csv('Churn_Modelling.csv')  #Import Dataset
X = dataset.iloc[:, 3:-1].values  #Remove Unnecasary Bloat Data From Rows/ Comluns (eg. Last Name)
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder   #Encode Non-Numeric Data (eg. Country)
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split    #Split data into training anf test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler  #Scale all values to be assesed in equal magnitude
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()   #Set up 3 Layers of 6 neurons of ANN
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  #Initialize output layer to giver probability
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)   #Specify batch size and amount of cycles

x = (str(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))*100)+"%")  #Input Test Case
x = x.replace("]","").replace("[","")  #Output result as probability
print("\nThe chance that the customer will leave in the next year is " + x)