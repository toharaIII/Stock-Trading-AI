import pandas as pd
import pickle
#time
#schedule

def loadDataframe(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        df=pickle.load(file)
    print(f"dataframe loaded from {filename}")
    return df

def appendSMA(df: pd.DataFrame, window: int) -> pd.DataFrame:
    lastCalculatedDay=df['SMA'].last_valid_index()
    if lastCalculatedDay is None:
        df['SMA']=df['close'].rolling(window=window, min_periods=1).mean()
    else:
        newSMA=df['close'].iloc[lastCalculatedDay+1:].rolling(window=window, min_periods=1).mean()
        df['SMA'].iloc[lastCalculatedDay+1]=newSMA
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
    lastCalculatedDay=df['RSI'].last_valid_index()
    if lastCalculatedDay is None:
        df=calculateRSI(df, window)
    else:
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
    lastCalculatedDay=df['Middle Band'].last_valid_index() if 'Middle Band' in df.columns() else None
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