from pandas import read_csv
from datetime import datetime
from pandas import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
	
def parser(x):
	return datetime.fromtimestamp(int(x))

series = read_csv('input/500004_500005.csv', header=None, parse_dates=[0], date_parser=parser, delimiter=',',names=['DateTime','AvgSpeed'])
series = series.set_index('DateTime')

#remove nans and outliers
mean=series.loc[series['AvgSpeed']<=35,'AvgSpeed'].mean()
series.loc[series.AvgSpeed>35,'AvgSpeed']=np.nan
series.fillna(mean, inplace=True)
series.fillna(series.mean(axis=0))

#Possibly difference the data
series = series.diff(periods=1)
series.dropna(inplace=True)

#plot ACF and PACF
series.plot()
pyplot.show()

'''autocorrelation_plot(series)
pyplot.show()'''

plot_acf(series).show()
pyplot.show()

plot_pacf(series, lags=50).show()
pyplot.show()
