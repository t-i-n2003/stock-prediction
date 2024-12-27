from vnstock import *
import pandas_ta as ta
from datetime import datetime
from functions import add_previous_and_next_7

today = datetime.today().strftime('%Y-%m-%d')
fpt = stock_historical_data(symbol="FPT", start_date="1999-01-01", end_date=today)
bbands = ta.bbands(fpt['close'], length=20, std=2)
fpt['BBL'] = bbands['BBL_20_2.0']
fpt['BBM'] = bbands['BBM_20_2.0']
fpt['BBU'] = bbands['BBU_20_2.0']
fpt['BBB'] = bbands['BBB_20_2.0']
fpt['BBP'] = bbands['BBP_20_2.0']
MACD = ta.macd(fpt['close'], fast=12, slow=26, signal=9)
fpt['MACD'] = MACD['MACD_12_26_9']
fpt['Signal'] = MACD['MACDs_12_26_9']
fpt['Histogram'] = MACD['MACDh_12_26_9']
fpt['Supertrend'] = ta.supertrend(fpt['high'], fpt['low'], fpt['close'], length=10, multiplier=3)['SUPERT_10_3.0']
fpt['RSI'] = ta.rsi(fpt['close'], length=14)

fpt = pd.DataFrame(fpt, columns=['time', 'BBL', 'BBM', 'BBU','Histogram'])
fpt = fpt.dropna()
fpt = add_previous_and_next_7(fpt, 7, 'Histogram')
fpt = add_previous_and_next_7(fpt, 7, 'BBU')
fpt = add_previous_and_next_7(fpt, 7, 'BBM')
fpt = add_previous_and_next_7(fpt, 7, 'BBL')
fpt = fpt.dropna()



