# Stock-Trading-AI
stock trading ai + other necessary features 

goal:
build an AI that can be given a series of stock ticker symbols and necessary historical data regarding said tickers train the AI on that historical data and then release it simulated money and let it make trades with equities and options.
We will also need to format the historical data into any indicators we deem necessary, such indicatiors are found below as a list, due to this we will also need to manipulate incoming data into said relevant indicators

indicator list:
  10 day simple moving average
  20 day simple moving average
  50 day simple moving average
  relative strenght index (RSI)
  Bollinger bands
  Ichimoku cloud (if time)

file list:
  dataFetch.py: initially going to collect historical data and then going to collect live data, appends both to a pandas dataframe
  indicators.py: intially creates indicators based on historical data passed in from dataFetch via pickle, but will then be called from dataFtech schedule function to updata indicators with live data pulled in
  model.py: the actual model which will execute decisions based on information provided by indicators, will look for good instances to buy and ood instances to sell for longs, shorts, potentially options if time
  positions.py: stores the closed and open positions that the model takes both so that the model can reference its current liquidity situation and also remember what positions it has open to then look for exit instances. This file will also be used by creators to evaluate live performance.
  

noteable libraries:
  requests: used for API calls over the internet
  alpha vantage: best option for getting stock data, API allows for users to request data over a specified period including live data, but no  more than 5 requests per minute and no more than 500 requests per day, but this isnt about high frequency trading so these limits should be fine
  pandas: used for dataframes so that we can create a dataset for historical data and then append live data brought in from alpha vantage API
  schedule: used for handling requests which are recurring based on some variable or run after a given period of time, i.e. on a schedule :D
  time: grants us access to current time in a variety of formats
  pickle: allows us to pass pandas dataframes between files in a more efficient manner than just using a CSV go between
  tensorflow and Keras: more popular for actual engineering vs Pytorch is primarily reserach focused
