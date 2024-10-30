import requests
import pandas as pd
import pickle
import schedule
import os
import time

"""
gets stock data for a given ticker symbol, ticker, which is passed in
uses alpha vantage to get all data from it IPO of the stock to right now
returns either a pandas dataframe of the stock data or an error message
"""
def getHistoricalData(ticker: str, apiKey: str):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={apiKey}'

    response=requests.get(url)
    data=response.json()

    if "Time Series (Daily)" in data:
        timeSeries=data['Time Series (Daily)']
        df=pd.DataFrame.from_dict(timeSeries, orient='index')
        df.columns=['open', 'high', 'low', 'close', 'volume']
        df.index=pd.to_datetime(df.index)
        df=df.sort_index(ascending=True)
        return df
    else:
        raise ValueError("Error fetching data from alpha vantage, check ticker or API key")

"""
gets the last 100 days of stock data for a passed in ticker symbol, ticker,
it checks to see if any of this data is newer than the newest data stored
in the pandas df and is so appends that data
"""
def updateData(ticker: str, apiKey: str, df: pd.DataFrame):
    lastDate=df.index.max().date()
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&interval=5min&apikey={apiKey}'
    
    response=requests.get(url)
    data=response.json()

    if "Time Series (Daily)" in data:
        timeSeries=data['Time Series (Daily)']
        newDf=pd.DataFrame.from_dict(timeSeries, orient='index')
        newDf.columns=['open', 'high', 'low', 'close', 'volume']
        newDf.index=pd.to_datetime(newDf.index)
        newDf = newDf[newDf.index.date > lastDate]

        if not newDf.empty:
            updatedDf=pd.concat([df, newDf]).sort_index(ascending=True)
            print("new data appended")
            return updatedDf
        else:
            print("No new data for this ticker")
            return df
    else:
        raise ValueError("Error fetching data from alpha vantage, check ticker or API key")

"""
saves a pandas dataframe to a pickle file for faster and easier inter file transfer
"""
def saveDfToPickle(df: pd.DataFrame, filename: str):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)
    print(f"DataFrame saved to {filename}")

"""
stores the list of stock tickers, fetchs teh alpha vantage api key, and stores the path to
the pickle data storage folder, has a for loop to go through each ticker in the list and
gets their historical data, and then schedules the update function for every hour
also schedules to update the pickle file after the update funciton is run
NOTE: apiKey is am enviornmental variable on my pc so it wont work on any other system
"""
def main():
    tickers=['META', 'AMZN', 'NFLX', 'GOOG']
    apiKey=os.getenv('ALPHA_VANTAGE_API_KEY') #this isnt going to work on any other system, due to lack of env var
    if not apiKey:
        raise ValueError("Alpha Vantage API key not found in environment variables.")
    folder='./data'

    os.makedirs(folder, exist_ok=True)

    for ticker in tickers:
        print(f"getting info for {ticker}")
        filePath=os.path.join(folder, f'{ticker}.pkl')

        if os.path.exists(filePath):
            with open(filePath, 'rb') as file:
                df=pickle.load(file)
                print(f"loaded data for {ticker}, fecthing new data")
                df=updateData(ticker, apiKey, df)
        else:
            print(f"getting historical data for {ticker}")
            df=getHistoricalData(ticker, apiKey)
            saveDfToPickle(df, filePath)
            print(f"fetched historical data for {ticker}, saved at {filePath}")
        
        print(df)
        time.sleep(12) #adjust based on number of api calls per day, limited to 25 per day rn
        schedule.every(1).hours.do(lambda t=ticker, a=apiKey, d=df, f=filePath: saveDfToPickle(updateData(t, a, d), f))
        print(f"scheduled updates for {ticker} every hour")

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()