import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf;
import keras

def main():
    dataPath='./data'
    modelPath='./trainedModel.h5'






    def load_model(model_path):
    # Remove the problematic time_major parameter
        def remove_time_major(config):
            if 'time_major' in config:
                del config['time_major']
            return config
    
    # Create custom objects dict with just the LSTM fix
        custom_objects = {
            'LSTM': lambda **kwargs: keras.layers.LSTM(**remove_time_major(kwargs))
        }
    
    # Load the model with the fix
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        return model

    # Use it like this:
    model = load_model(modelPath)






    stockData={}

    for file in os.listdir(dataPath):
        if not file.endswith('.pkl'):
            continue
        ticker=file.replace('.pkl','') #collect ticker symbols form file names
        with open(os.path.join(dataPath, file), 'rb') as file:
            df=pickle.load(file)
            stockData[ticker]=df
        
    commonDates=set.intersection(*(set(df.index) for df in stockData.values()))
    commonDates=list(commonDates)
    for ticker in stockData:
        stockData[ticker]=stockData[ticker].loc[commonDates].sort_index()

    cash=10000
    portfolioCash={ticker: cash/len(stockData) for ticker in stockData} #allocates 1/x of the cash for each stock we will be trading
    portfolioPositions={ticker: None for ticker in stockData} #tracks current position for each stock
    portfolioShares={ticker: 0 for ticker in stockData} #tracks number of shares in active position for each stock
    tradeLog=[] #history of all trades made

    SEQUENCE_LENGTH = 60

    for dateIdx, date in enumerate(sorted(commonDates)):
        for ticker, df in stockData.items():
            print(ticker)
            if dateIdx < SEQUENCE_LENGTH:  # Skip until we have enough history
                continue

            sequence=df.iloc[dateIdx - SEQUENCE_LENGTH:dateIdx]
            row=df.loc[date] #get the row for the corresponding date
            closingPrice=row['close']

            inputData=np.array([sequence.values], dtype=np.float32)
            predictedPrice=model.predict(inputData, verbose=0)[0]

            position=portfolioPositions[ticker]
            cash=portfolioCash[ticker]
            shares=portfolioShares[ticker]

            if position=='long':
                if predictedPrice<closingPrice: #exit long position
                    portfolioCash[ticker]+=shares*closingPrice
                    portfolioShares[ticker]=0
                    portfolioPositions[ticker]=None
                    tradeLog.append((date, ticker, 'Exit Long', closingPrice))
                else: #maintain long position
                    tradeLog.append((date, ticker, 'Hold Long', closingPrice))
            
            elif position=='short':
                if predictedPrice>closingPrice: #exit short position
                    portfolioCash[ticker]-=shares*closingPrice
                    portfolioShares[ticker]=0
                    portfolioPositions[ticker]=None
                    tradeLog.append((date, ticker, 'Exit Short', closingPrice))
                else: #hold short position
                    tradeLog.append((date, ticker, 'Hold Short', closingPrice))
            
            elif position is None:
                if predictedPrice>closingPrice: #enter long position
                    shares=cash//closingPrice
                    portfolioCash[ticker]-=shares*closingPrice
                    portfolioShares[ticker]=shares
                    portfolioPositions[ticker]='long'
                    tradeLog.append((date, ticker, 'Enter Long', closingPrice))
                elif predictedPrice < closingPrice:
                    shares=cash//closingPrice
                    portfolioCash[ticker]+=shares*closingPrice
                    portfolioShares[ticker]=shares
                    portfolioPositions[ticker]='Short'
                    tradeLog.append((date, ticker, 'Enter Short', closingPrice))

    finalPortfolioValue=0

    for ticker in stockData:
        cash=portfolioCash[ticker]
        position=portfolioPositions[ticker]
        shares=portfolioShares[ticker]
        lastPrice=stockData[ticker].iloc[-1]['close']

        finalPortfolioValue+=cash
        if position=='long':
            finalPortfolioValue+=shares*lastPrice
        elif position=='short':
            finalPortfolioValue-=shares*lastPrice

    print(f"Final Portfolio Value: {finalPortfolioValue}")
    print(f"Trades: {tradeLog}")


if __name__ == "__main__":
    main()