## Missing Time Series Data Univariate
import pandas as pd
import numpy as np

rowsOfData = int(input())

Data = []
while(rowsOfData):
    data = input()
    Data.append(data.split())
    rowsOfData -= 1
DF = pd.DataFrame(Data)
DF.columns = ['date', 'time', 'value']
del DF['date']
del DF['time']

DF.value = pd.to_numeric(DF.value, errors = 'coerce')

missingIndexList = DF.loc[pd.isna(DF['value']), :].index
DF.interpolate(method = 'polynomial', order = 2, inplace = True)

for i in missingIndexList:
    print(DF.at[i, 'value'])


# Forecast Data Univariate -- No seasonal No Trend
# Enter your code here. Read input from STDIN. Print output to STDOUT

# Dependencies
import numpy as np
import pandas as pd
import sklearn

testCases = int(input())
Arr = []
while(testCases):
    Arr.append(int(input()))
    testCases -= 1

# Predict next 30 days -- forecast

# Conversion to numpy 1d array
train = np.array(Arr)

# Smoothing the series -- Moving averages (5 day moving average)
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=int)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

train_smoothed = moving_average(train, 5)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_smoothed, order = (2, 1, 3))
model_fit = model.fit(disp=0)

forecast = model_fit.forecast(steps = 30)[0]

forecast = list(forecast)

for future in forecast:
    print(int(future))

# Time Series ForeCasting -- Seasonal and Trend
# Enter your code here. Read input from STDIN. Print output to STDOUT

# Dependencies
import numpy as np
import pandas as pd
import sklearn

testCases = int(input())
Arr = []
while(testCases):
    Arr.append(float(input()))
    testCases -= 1

# Predict next 30 days -- forecast
Arr = Arr + Arr
# Conversion to numpy 1d array
train = np.array(Arr)

# Smoothing the series -- Moving averages (5 day moving average)
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

train_smoothed = moving_average(train, 5)

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train, order=(0,0,0), seasonal_order=(0,1,0,24))
model_fit = model.fit(disp=0)

forecast = model_fit.forecast(steps = 30)
print(forecast)

forecast = list(map(float, forecast))

for future in forecast:
    print(round((float(future)), 2))

## Time series forecasting -- IF real values also given (test also given)
def rolling_forecast(train, test, order, season):
    history = [x for x in train]
    model = SARIMAX(history, order= order, seasonal_order= season)
    model_fit = model.fit(disp=False)
    predictions = []
    results = {}
    yhat = model_fit.forecast()[0]

    predictions.append(yhat)
    history.append(test[0])
    for i in range(1, len(test)):
        model = SARIMAX(history, order= order, seasonal_order= season)
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        obs = test[i]
        history.append(obs)
    return predictions

