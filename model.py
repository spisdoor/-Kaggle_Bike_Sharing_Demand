import pandas as pd
import numpy as np
# from sklearn import metrics, cross_validation, preprocessing

train = pd.read_csv('train1.csv', index_col=None, na_values=['NA'])
test = pd.read_csv('test1.csv', index_col=None, na_values=['NA'])

X = train[['atemp', 'holiday', 'humidity', 'temp', 'windspeed', 'workingday', 'year', 'month', 'day', 'hour', 'season', 'weather_1', 'weather_2', 'weather_3', 'weather_4']]
y = train['count']
data_test = test[['atemp', 'holiday', 'humidity', 'temp', 'windspeed', 'workingday', 'year', 'month', 'day', 'hour', 'season', 'weather_1', 'weather_2', 'weather_3', 'weather_4']]

def rmsle(predicted, actual):
  p = np.array([np.log(i + 1) for i in predicted])
  a = np.array([np.log(i + 1) for i in actual])
  return np.sqrt(np.mean((p - a) ** 2))

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X, np.log(y))
# predict = rf.predict(X)
# print ('rf: ', rmsle(np.exp(np.log(y)), np.exp(predict)))

# for name, importance in zip(X.columns, rf.feature_importances_):
#   print (name, importance)

# print ('--------------------')

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=500)
gbr.fit(X, np.log(y))
predict = gbr.predict(X)
print ('gbr: ', rmsle(np.exp(np.log(y)), np.exp(predict)))

for name, importance in zip(X.columns, gbr.feature_importances_):
  print (name, importance)

model = gbr
model.fit(X, np.log(y))
predictions = model.predict(data_test)
result = pd.DataFrame({'datetime': test['datetime'], 'count':[max(0, x) for x in np.exp(predictions)]})
result.to_csv('result.csv', index=False)
