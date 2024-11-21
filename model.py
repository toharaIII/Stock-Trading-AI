import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf; tf.keras
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

def preprocessData(filePath, sequenceLength=60):
    #need to make this a loop to get all of them
    df=pd.read_pickle(r'./data/META.pkl')
    # Convert columns to numeric, invalid parsing will be set as NaN
    df[['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 
        'Middle Band', 'Upper Band', 'Lower Band']] = df[['open', 'high', 'low', 'close', 
        'volume', 'SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 'Middle Band', 'Upper Band', 
        'Lower Band']].apply(pd.to_numeric, errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    #MinMaxScaler is an object that allows you to normalize data 
    #(set data max value to 1 and min to 0 with all others falling withint 0-1 proportionally)
    scaler=MinMaxScaler()
    #applying scaler to all columns in a given stock df
    dfScaled=scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume', 
                                     'SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 
                                     'Middle Band', 'Upper Band', 'Lower Band']])

    inputs, output=[],[]
    for i in range(sequenceLength, len(dfScaled)):
        inputs.append(dfScaled[i-sequenceLength:i]) #slicing every sequence number of days into an element in inputs
        output.append(dfScaled[i,3]) #i believe closing price is third column
    return np.array(inputs), np.array(output), scaler #return scaler to reverse scaling for graphing

def buildModel(inputShape):
    model=Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=inputShape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2)) #20% of all neurons are 'dropped' (set to 0) to prevent individual neuron dependence
    model.add(Dense(1)) #output is 1 value
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model;

def main():
    filePath='./data'
    sequenceLength=60
    allInputs=[]
    allOutputs=[]

    for file in os.listdir(filePath):
        filePathJoined=os.path.join(filePath, file)
        if not file.endswith('.pkl'):
            continue

        print(f"processing file: {file}")
        inputs, output, scaler=preprocessData(filePathJoined, sequenceLength)
        allInputs.append(inputs)
        allOutputs.append(output)

    allInputs=np.concatenate(allInputs, axis=0)
    allOutputs=np.concatenate(allOutputs, axis=0)

    xTrain, xTest, yTrain, yTest = train_test_split(inputs, output, test_size=0.2, shuffle=False)
    model=buildModel((xTrain.shape[1], xTrain.shape[2]))
    history=model.fit(xTrain, yTrain, epochs=50, batch_size=32, validation_data=(xTest, yTest))
    model.save('./trainedModel')
    
    yPred=model.predict(xTest)

    plt.plot(yTest, label="Actual Prices")
    plt.plot(yPred, label="Predicted Prices")
    plt.legend()
    plt.title(f"Predicted vs. Actual Closing Prices for {file}")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.show()

if __name__ == "__main__":
    main()