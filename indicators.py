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
    if f'SMA_{window}' not in df.columns:
        df[f'SMA_{window}']=pd.Series(dtype=float)

    lastCalculatedDay=df[f'SMA_{window}'].last_valid_index()
    if lastCalculatedDay is None:
        df[f'SMA_{window}']=df['close'].rolling(window=window, min_periods=1).mean()
    else:
        if lastIndex + 1 >= len(df):
            df.loc[len(df)] = [None] * len(df.columns)
        lastIndex=df.index.get_loc(lastCalculatedDay)
        newSMA=df['close'].iloc[lastIndex+1:].rolling(window=window, min_periods=1).mean()
        df.loc[df.index[lastIndex+1], f'SMA_{window}']=newSMA
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
    if 'RSI' not in df.columns:
        df['RSI']=pd.Series(dtype=float)

    lastCalculatedDay=df['RSI'].last_valid_index()
    if lastCalculatedDay is None:
        df=calculateRSI(df, window)
    else:
        if lastIndex + 1 >= len(df):
            df.loc[len(df)] = [None] * len(df.columns)
        delta=df['close'].diff(1)
        gain=delta.where(delta > 0,0)
        loss=delta.where(delta < 0,0)

        new_avg_gain=gain.iloc[lastCalculatedDay+1:].rolling(window=window, min_periods=1).mean()
        new_avg_loss=loss.iloc[lastCalculatedDay+1:].rolling(window=window, min_periods=1).mean()
        RS=new_avg_gain/new_avg_loss
        new_RSI=100-(100/(1+RS))

        df['RSI'].iloc[lastCalculatedDay+1:]=new_RSI #appending to existing df
    return df

def appendBollingerBands(df: pd.DataFrame, window: int, multiplier: float=2.0) -> pd.DataFrame:
    lastCalculatedDay=df['Middle Band'].last_valid_index() if 'Middle Band' in df.columns else None
    if lastCalculatedDay is None:
        df['Middle Band']=df['close'].rolling(window=window, min_periods=1).mean()
        rollingStd=df['close'].rolling(window=window, min_periods=1).std()

        df['Upper Band']=df['Middle Band']+(rollingStd*multiplier)
        df['Lower Band']=df['Middle Band']-(rollingStd*multiplier)
    else:
        newPrices=df['close'].iloc[lastCalculatedDay+1:]
        newMiddleBand=newPrices.rolling(window=window, min_periods=1).mean()
        newRollingStd=newPrices.rolling(window=window, min_periods=1).std()

        newUpperBand=newMiddleBand+(newRollingStd*multiplier)
        newLowerBand=newMiddleBand-(newRollingStd*multiplier)

        df['Middle Band'].iloc[lastCalculatedDay+1]=newMiddleBand #appending to existing df
        df['Upper Band'].iloc[lastCalculatedDay+1]=newUpperBand
        df['Lower Band'].iloc[lastCalculatedDay+1]=newLowerBand
    return df

def processIndicators(filename: str, smaWindowOne: int, smaWindowTwo: int, smaWindowThree: int, rsiWindow: int, bollingerWindow: int, multiplier: float):
    df=loadDataframe(filename)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    print("sma 1")
    df=appendSMA(df, smaWindowOne)
    print("sma 2")
    df=appendSMA(df, smaWindowTwo)
    print("sma 3")
    df=appendSMA(df, smaWindowThree)
    #print(df)
    df=appendRSI(df, rsiWindow)
    df=appendBollingerBands(df, bollingerWindow, multiplier)

    with open(filename, 'wb') as file:
        pickle.dump(df, file)
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
                        if lastModTime.get(file) is None or currentModTime > lastModTime:
                            print("new data uploaded, writing corresponding indicators...")
                            processIndicators(filePath, smaWindowOne, smaWindowTwo, smaWindowThree, rsiWindow, bollingerWindow, multiplier)
                            lastModTime[file]=currentModTime
                    except Exception as e:
                        print(f"error ocurred:, {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()