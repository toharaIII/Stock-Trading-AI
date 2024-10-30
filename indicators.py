import pandas as pd
import pickle
import os
import time

def loadDataframe(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        df=pickle.load(file)
    print(f"dataframe loaded from {filename}")
    return df

def appendSMA(df: pd.DataFrame, window: int) -> pd.DataFrame:
    sma_column = f'SMA_{window}'
    if sma_column not in df.columns:
        df[sma_column] = pd.Series(dtype=float)

    last_calculated_day = df[sma_column].last_valid_index()
    
    # Calculate SMA from the beginning if it's not present
    if last_calculated_day is None:
        df[sma_column] = df['close'].rolling(window=window, min_periods=1).mean()
    else:
        last_index = df.index.get_loc(last_calculated_day)
        
        # Ensure we only calculate for missing SMA values
        new_sma = df['close'].iloc[last_index:].rolling(window=window, min_periods=1).mean()
        
        # Assigning the new SMA values to appropriate rows
        df.loc[df.index[last_index + 1]:, sma_column] = new_sma[1:]

    return df

def calculateRSI(df: pd.DataFrame, window: int=14) -> pd.DataFrame:
    delta=df['close'].diff(1)
    gain=delta.where(delta > 0,0)
    loss=delta.where(delta < 0,0)

    avgGain=gain.rolling(window=window, min_periods=1).mean()
    avgLoss=loss.rolling(window=window, min_periods=1).mean()
    rs=avgGain/avgLoss

    df[f'RSI_{window}']=100-(100/(1+rs)) #appending to existing df
    return df

def appendRSI(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rsi_column = f'RSI_{window}'
    if rsi_column not in df.columns:
        df[rsi_column] = pd.Series(dtype=float)

    last_calculated_day = df[rsi_column].last_valid_index()

    if last_calculated_day is None:
        df = calculateRSI(df, window)
    else:
        last_index = df.index.get_loc(last_calculated_day)
        
        # Calculate gains and losses only for the range that needs RSI updates
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)  # make loss positive

        # Calculate new average gain and loss, RS, and RSI values
        new_avg_gain = gain.iloc[last_index + 1:].rolling(window=window, min_periods=1).mean()
        new_avg_loss = loss.iloc[last_index + 1:].rolling(window=window, min_periods=1).mean()
        rs = new_avg_gain / new_avg_loss
        new_rsi = 100 - (100 / (1 + rs))
        
        # Append new RSI values to the DataFrame
        df.loc[df.index[last_index + 1]:, rsi_column] = new_rsi

    return df


def appendBollingerBands(df: pd.DataFrame, window: int, multiplier: float = 2.0) -> pd.DataFrame:
    if 'Middle Band' not in df.columns:
        df['Middle Band'] = pd.Series(dtype=float)
        df['Upper Band'] = pd.Series(dtype=float)
        df['Lower Band'] = pd.Series(dtype=float)

    last_calculated_day = df['Middle Band'].last_valid_index()

    if last_calculated_day is None:
        df['Middle Band'] = df['close'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=window, min_periods=1).std()

        df['Upper Band'] = df['Middle Band'] + (rolling_std * multiplier)
        df['Lower Band'] = df['Middle Band'] - (rolling_std * multiplier)
    else:
        last_index = df.index.get_loc(last_calculated_day)
        
        # Compute new Bollinger Bands for the range after the last calculated day
        new_prices = df['close'].iloc[last_index + 1:]
        new_middle_band = new_prices.rolling(window=window, min_periods=1).mean()
        new_rolling_std = new_prices.rolling(window=window, min_periods=1).std()

        new_upper_band = new_middle_band + (new_rolling_std * multiplier)
        new_lower_band = new_middle_band - (new_rolling_std * multiplier)

        # Append new Bollinger Bands to the DataFrame
        df.loc[df.index[last_index + 1]:, 'Middle Band'] = new_middle_band
        df.loc[df.index[last_index + 1]:, 'Upper Band'] = new_upper_band
        df.loc[df.index[last_index + 1]:, 'Lower Band'] = new_lower_band

    return df

def processIndicators(filename: str, smaWindowOne: int, smaWindowTwo: int, smaWindowThree: int, rsiWindow: int, bollingerWindow: int, multiplier: float):
    df=loadDataframe(filename)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    print(df)
    print("sma 1")
    df=appendSMA(df, smaWindowOne)
    print("sma 2")
    df=appendSMA(df, smaWindowTwo)
    print("sma 3")
    df=appendSMA(df, smaWindowThree)
    print("rsi")
    df=appendRSI(df, rsiWindow)
    print("bollinger")
    df=appendBollingerBands(df, bollingerWindow, multiplier)
    print(df)

    with open(filename, 'wb') as file:
        pickle.dump(df, file)
        print("dumped")
    print(f"indicators written/updated for {filename}")

def main():
    filepath='./data'
    smaWindowOne=10
    smaWindowTwo=20
    smaWindowThree=50
    rsiWindow=14
    bollingerWindow=20
    multiplier=2.0
    lastModTime={}

    while True:
        for file in os.listdir(filepath):
            print(f"file: {file}")
            filePath=os.path.join(filepath, file)
            print(f"filepath: {filePath}")  
            if os.path.exists(filePath):

                    try:
                        currentModTime=os.path.getmtime(filePath)
                        print(lastModTime)
                        if lastModTime.get(file) is None or currentModTime > lastModTime.get(file):
                            print("new data uploaded, writing corresponding indicators...")
                            processIndicators(filePath, smaWindowOne, smaWindowTwo, smaWindowThree, rsiWindow, bollingerWindow, multiplier)
                            lastModTime[file]=currentModTime
                    except Exception as e:
                        print(f"error ocurred:, {e}")
                        break
        time.sleep(60)

if __name__ == "__main__":
    main()