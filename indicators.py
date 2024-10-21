import pandas as pd
import pickle
#time
#schedule

def loadDataframe(filename: str) -> pd.DataFrame:
    with open(filename, 'rb') as file:
        df=pickle.load(file)
    print(f"dataframe loaded from {filename}")
    return df