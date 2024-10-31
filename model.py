import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#need to make this a loop to get all of them
df=pd.read_pickle(r'/data/stock')

 #MinMaxScaler is an object that allows you to normalize data 
 #(set data max value to 1 and min to 0 with all others falling withint 0-1 proportionally)
scaler=MinMaxScaler()
#applying scaler to all columns in a given stock df
dfScaled=scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', 
                                     'SMA_10', 'SMA_20', 'SMA_50', 'RSI', 
                                     'BB_Mid', 'BB_Low', 'BB_High']]) #names off memory, need to double check
"""
ok heres whats happening in this function:
we have a single dataframe and a sequence length as the inputs
a sequence is a period of days which defautls to 60
our inputs is a list that will take a sequence number of rows ofo data from our dataframe and append them as an element in the list
our output list will append the closing value of the day after the last day in the sequence so sequence i sdays 0-59 output is closing on day 60
reason being that the ai will look at all data in a given index in inputs and believe that some components of it is leading to the output value being what it is
and will try to determine what if any correslations can be made, thus determining a trading strategy and when to act based off said strategy
"""
def createSequences(Data: pd.DataFrame, sequenceLength=60) -> pd.DataFrame:
    inputs, output=[],[]
    for i in range(sequenceLength, len(Data)):
        inputs.append(Data[i-sequenceLength:i]) #slicing every sequence number of days into an element in inputs
        output.append(Data[i:3]) #i believe closing price is third column
    return np.array(inputs), np.array(output)
sequenceLength=60
inputs, output=createSequences(dfScaled, sequenceLength)

"""
regarding below section of code:
we are using a sequential model, which means that it feeds the outputs from each layer directly into the next layer
we have 5 layers in total, 2 LSTM layers, 2 dropout layers and 1 dense layer
LSTM layer: (long short term memory) retains some data from each element in our input array that was previously passed in
this allows it access certain pieces of past sequence data when determining the outputs for the current iteration (within the same training epoch)
dropout layer: does no atual calculations instead it will randomly set 20% of our 50 neurons in each layer to 0 activitions and therfore 0 outputs
this is useful since it prevents decision making from becoming dependent on a few arbitarily selected neurons instead of using the entire apparatus
dense: all 50 neurons from the prior layer output into a single neuron which makes a calculation and outputs the predicted price
"""
model=Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2)) #20% of all neurons are 'dropped' (set to 0) to prevent individual neuron dependence
model.add(Dense(1)) #output is 1 value

model.compile(optimizer='adam',loss='mean_squared_error')

#unsure of following components
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
# You can inverse the scaling if needed to get the price scale back
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))