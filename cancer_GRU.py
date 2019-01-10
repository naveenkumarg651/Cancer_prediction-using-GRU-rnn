import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,GRU,Dropout,LSTM
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import roc_auc_score

train=pd.read_csv('G:/my_past/mypython/cancer.csv')
data=train.iloc[:400,:-1].fillna(0).values
test_data=train.iloc[400:,:-1].fillna(0).values
test_data=np.expand_dims(test_data,axis=2)
data=np.expand_dims(data,axis=2)
targets=train.iloc[:400,-1].values
test_targets=train.iloc[400:,-1].values

input_=Input(shape=(30,1))
rnn=LSTM(512,activation='sigmoid',input_shape=(30,1))(input_)
rnn=Dropout(0.6)(rnn)
x=Dense(1,activation='sigmoid')(rnn)


model=Model(input_,x)
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=Adam(lr=0.001))
r=model.fit(data,targets,batch_size=64,epochs=1,validation_split=0.2)

plt.plot(r.history['acc'],label="accuracy")
plt.plot(r.history['val_acc'],label="val_accuracy")
plt.legend()
plt.show()

plt.plot(r.history['loss'],label="loss")
plt.plot(r.history['val_loss'],label="val_loss")
plt.legend()
plt.show()
predict=model.predict(test_data)
print(np.shape(test_targets),np.shape(predict))
print("roc_auc_score: ",roc_auc_score(test_targets,predict))

