import numpy as np 
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler


# Set to Max as there're too many columns in dateset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Import data for differentiation
df_train = pd.read_csv(r'C:\Users\jezec\Desktop\MBA\DataScience\PDD\data\cleansed_train.csv')
df_test = pd.read_csv(r'C:\Users\jezec\Desktop\MBA\DataScience\PDD\data\cleansed_test.csv')

df_train=df_train.replace({np.nan:-1})
df_test=df_test.replace({np.nan:-1})

for i in df_train.columns:
    if df_train[i].dtype=='object':
        print (i)

category_list=[]
for i in df_train.columns:
    if df_train[i].dtype=='object':
        category_list.append(i)

for i in category_list:
    lbl = preprocessing.LabelEncoder()
    df_train[i] = lbl.fit_transform(list(df_train[i].values))
    df_test[i] = lbl.fit_transform(list(df_test[i].values))

x_train=df_train.drop('target',axis=1)
y_train=df_train['target']
x_test=df_test.drop('target',axis=1)
y_test=df_test['target']

x_train=x_train.astype(np.float32)
y_train=y_train.astype(np.float32)


model = Sequential()
model.add(Dense(units=1024, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

scaler = StandardScaler().fit(x_train)
x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)

model.fit(x_train, y_train, epochs=80, batch_size=128,
          callbacks = [EarlyStopping(monitor='loss', patience=20)],
          validation_data=(x_test,y_test))
y_pre = model.predict_proba(x_test)
scores_train = roc_auc_score(y_train,model.predict_proba(x_train))
scores = roc_auc_score(y_test,y_pre)


print ("\nnnModel Report")
print ("AUC Score (Train): %f" %scores_train)
print ("AUC Score (Test): %f" %scores)