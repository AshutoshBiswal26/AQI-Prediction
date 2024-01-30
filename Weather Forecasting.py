import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
import calendar
from matplotlib import pyplot
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

user = 'root'
password = '2607'
db = 'ashutosh'

engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")
df = pd.read_excel(r"C:/Users/ashut/OneDrive/Documents/AQI/Air.xlsx")
df.to_sql('AQI', con = engine, if_exists = 'replace', index = False, chunksize = 25000)

sql = 'select * from AQI'
AQI = pd.read_sql_query(sql, con = engine)

Train = AQI.head(903)
Test = AQI.tail(900)

Test.to_excel('C:/Users/ashut/OneDrive/Documents/AQI/Test_Air.xlsx')
import os 
os.getcwd()

df = pd.read_excel(r'C:/Users/ashut/OneDrive/Documents/AQI/Test_Air.xlsx', index_col = 0)
tsa_plots.plot_acf(AQI.co, lags = 12)
tsa_plots.plot_pacf(AQI.aqi, lags = 12)

model1 = ARIMA(Train.aqi, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())

start_index = len(Train)
start_index
end_index = start_index + 222
forecast_test = res1.predict(start = start_index, end = end_index)
print(forecast_test)

rmse_test = sqrt(mean_squared_error(Test.aqi, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

pyplot.plot(Test.aqi)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()

import pmdarima as pm
help(pm.auto_arima)

ar_model = pm.auto_arima(Train.aqi, start_p = 0, start_q = 0,
                      max_p = 12, max_q = 12, # maximum p and q
                      m = 12,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = True,   # Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)

model = ARIMA(Train.aqi, order = (1, 1, 1))
res = model.fit()
print(res.summary())

start_index = len(Train)
end_index = start_index + 221
forecast_best = res.predict(start = start_index, end = end_index)
print(forecast_best)

rmse_best = sqrt(mean_squared_error(Test.aqi, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
# plot forecasts against actual outcomes
pyplot.plot(Test.aqi)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)

res1.save("model.pickle")

from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

start_index = len(AQI)
end_index = start_index + 221
forecast = model.predict(start = start_index, end = end_index)

print(forecast)
