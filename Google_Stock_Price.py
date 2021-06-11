# -*- coding: utf-8 -*-
#Recurrent Neural Network

#Part 1 - Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training set(training RNN only on training set)
#in keras only numpy arrays can be input of RNN
#import as dataframe,select right column i.e open google stock price and make it a numpy 
                                                                                 #array
#open column is stock price at beginning of financial day as we are going to preict the same
data_train = pd.read_csv("Google_Stock_Price_Train.csv")
#input data of RNN
#we cannot take only 1 as we want array but it creates a single vector,we need
                                      # to take range 1:2
training_set = data_train.iloc[:,1:2].values#in a range,2 is excluded

#Feature Scaling
#If building RNN model and especially if we use sigmoid function in the output 
                                       #layer then Normalisation is recommended
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))#(0,1) is the range of normalised values
training_set_scaled = sc.fit_transform(training_set)#fit will get the min 
                          #and max of the data and transform will calculate the scaled data

#Creating a data structure with 60 timesteps and 1 output
#60 timesteps means that at each time 't' the RNN is going to look at 60 stock prices  
         #before time 't' i.e the stock prices between 60 days before time t and time t 
         #and based on the trends it is capturing during those 60 previous time steps,
         #model will predict stock price at time 't+1'
#60 time steps(60 previous financial days) - 3 months(20 financial days in a month)
#at each,we will look at 3 previous month stock prices to predict for the next day
#to predict for time 't+1',we will have 60 timesteps as input and 1 output
#wrong prediction of time steps can lead to overfitting(found by trial and error)
#create 2 entities.one entity is xtrain(input) and the other is ytrain(output) 
#for each observation,xtrain - 60 previous stock prices and ytrain - stock price of
                                                              #next financial day 
xtrain = []
ytrain = []
for i in range(60,1258):#for each i, we will get i-60 to i
    xtrain.append(training_set_scaled[i-60:i,0])#[i-60,i] are the previous 60 values 
                                                  #and '0' first column of 
          #training_set_scaled indicates the values which are to be taken for timestep 
    ytrain.append(training_set_scaled[i,0])
#xtrain and ytrain are lists so we have to transform them to numpy array 
                                             #such that RNN model can accept
xtrain,ytrain = np.array(xtrain),np.array(ytrain)

#Reshaping the data(we can add more indicators like 'open' for better prediction)
#to be compatible with the input format of RNN,reshaping is used
xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
                    #1st arg-array to be reshaped,2nd arg-structure of the reshaped array
                                  #to get batch size - xtrain.shape[0](number of rows),
                                  #to get timesteps - xtrain.shape[1](number of columns)
                                  #1 indicates 1 dimension which is 'open'
                                  #3D structure is created

#Part 2 - Building the RNN

from keras.models import Sequential#to create NN object representing sequence of layers
from keras.layers import Dense#to add output layer
from keras.layers import LSTM#to  add LSTM layers
from keras.layers import Dropout#to add dropout regularisation(avoids overfitting)

#Initialising the RNN
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(xtrain.shape[1],1)))
                     #LSTM object is created
                     #as capturing the trends of the stock price is complex,model must 
                     #have high dimensionality(more neurons),we will get it through stacking 
                     #of LSTMs(multiple LSTMs) and also by adding more units/neurons in one layer
                     #units - number of LSTM cells/memory units/neurons,
                     #return sequences(true)-stacked LSTM else false(default value)
#input shape - shape of the input xtrain(3D),we have to include only the last 2 args 
#and not the 1st because it will be automatically taken into account
regressor.add(Dropout(0.2))
#rate - nnumber of neurons to drop ignore in the layers to perform regularisation
                           #20% of neurons are ignored during forward and 
                           #backward propagation(one iteration in training)20% of 50 is 10
                           #10 neurons are ignored in each iteration

#Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True))
#no need to specify input shape as units of previous layer says that there are 
#50 neurons in the previous layer
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))#fully connect to the 4th LSTM layer,so dense class
#1 output(stock price at time 't+1') so 1 dimension in output layer

#Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
             #rmsprop is recommended for RNN but 
             #adam is preferred as it always performs relevant updates of the weights 
             #which makes it a safe and powerful optimizer
             #we are predicting a continuous value so it is a regression problem,
             #and error can only be measured by MSE

#Fitting the RNN to the training set
regressor.fit(xtrain,ytrain,epochs=100,batch_size=32)
#instead of updating the weights after every stock price forward propagating the 
#neural network and then back propagating,it is done after every 32 stock prices 

#Part 3 - Making the predictions and visualising the results

#Getting the real stockprice of 2017
data_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = data_test.iloc[:,1:2].values

#Getting the predicted stockprice of 2017
data_total = pd.concat((data_train['Open'],data_test['Open']),axis=0)
#vertical concatenation(merging same columns) - axis=0
#horizontal concatenation(merging same rows) - axis=1

inputs = data_total[len(data_train)-len(data_test)-60:].values
                        #1278      -    20 = 1258(index of 1st sample of test data)
#finds previous 60 stock prices for every sample in test data starting from 1st to 20th 
#Input Data :
#we are predicting the value at time 't+1' from 60 previous stock prices(3 months data)
#for predicting the values in test set,we even require january 2017 values(test data),
#we need to merge test and train set.we need to merge original dataframes and then scale at once.
#we have to scale this input data as the RNN model is trained on scaled data

inputs = inputs.reshape(-1,1)#for a proper shape of the array i.e inputs in lines with 1 column
#-1 : unknown rows , 1 : 1 column
 
inputs = sc.transform(inputs)
#we need to scale the inputs and not the actual test values
 
xtest = []#as we are only predicting and not training so no ytest

for i in range(60,80):#60+20(test set)
    xtest.append(inputs[i-60:i,0]) 
xtest = np.array(xtest)
xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1)) 
#Creating the 3D format of the input array such that it is suitable for 
#training and for prediction 

predicted_stock_price = regressor.predict(xtest)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)#inverse scaling 

#Visualising the results(model couldnot react to fast non linear changes)
plt.plot(real_stock_price,color='green',label='Real Google Stock Price(Jan 2017)')
plt.plot(predicted_stock_price,color='red',label='Predicted Google Stock Price(Jan 2017)')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
