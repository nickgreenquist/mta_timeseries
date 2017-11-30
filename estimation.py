from pandas import read_csv
from datetime import datetime
from pandas import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot
import itertools
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(df, arima_order):
	X = df.values
	size = int(len(X) * .8)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		#model = SARIMAX(history, order=arima_order, seasonal_order=(1,1,1,144))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)
	# plot
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.title(str(arima_order))
	pyplot.show()

	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

def fit_models(dataset, p_values, d_values, q_values):
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					print("\n\n\n\n")
					model = ARIMA(dataset, order=order)
					model_fit = model.fit()
					#print(model_fit.summary())
					print('ARIMA{} - AIC:{}'.format(param, model_fit.aic))
					with open("results.txt", 'a') as f:
						f.write('\nARIMA{} - AIC:{}'.format(param, model_fit.aic))
					f.close()
				except:
					continue

	# plot residual errors
	'''residuals = pd.DataFrame(model_fit.resid)
	residuals.plot()
	pyplot.show()
	residuals.plot(kind='kde')
	pyplot.show()
	print(residuals.describe())'''

def parser(x):
	return datetime.fromtimestamp(int(x))

series = read_csv('input/500004_500005.csv', header=None, parse_dates=[0], date_parser=parser, delimiter=',',names=['DateTime','AvgSpeed'])
series = series.set_index('DateTime')

#remove nans and outliers
mean = series.loc[series['AvgSpeed']<=35,'AvgSpeed'].mean()
series.loc[series.AvgSpeed>35,'AvgSpeed']=np.nan
series.fillna(mean, inplace=True)
series.fillna(series.mean(axis=0))

#series = series[:1200]

# seasonal difference
#series = series.diff(144)
# trim off the first year of empty data
#series = series[144:]


# GRID SEARCH
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
#seasonal_pdq = [(x[0], x[1], x[2], 144) for x in list(itertools.product(p, d, q))]


#evaluate_models(series, p_values, d_values, q_values)
#evaluate_models(series, [0], [1], [0])

#create an ARIMA model using the above orders
#evaluate_arima_model(series, (2, 1, 1))

#fit_models(series, p_values, d_values, q_values)
fit_models(series, pdq)


#DO NOT DELETE
#Best for 500004_500005:288 - (2,1,1)
#Best for 500004_500005:576 - (0,0,2)
#Best for 500004_500005:3074 - (8,0,1)