import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

housing = pd.read_csv('albanyrent.csv')
print(housing.head())
print(housing.info())
print(housing.describe())

housing.drop(['id','sqft_living','sqft_lot','waterfront','view','grade','sqft_above','sqft_basement','sqft_living15','sqft_lot15'],axis=1,inplace=True)
print(housing.head())

#sns.pairplot(housing)
#plt.show()

sns.distplot(housing['price'])
plt.show()

sns.heatmap(housing.corr(),cmap='BrBG')
plt.show()

x = housing[['bedrooms','bathrooms','floors','condition','lat','long']]
y = housing['price']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train,y_train)

bedroom_input = input('Please enter the required number of bedrooms.')
bathroom_input = input('Please enter the required number of bathrooms.')
condition_input = input('Please enter the required condition of the house.')
lat_input = input('Please enter the latency of the required house place.')
long_input = input('Please enter the longitude of the required house place.')
floors_input = input('Please enter the required number of floors.')
#yr_renovated_input = input('Please enter the year the house was renovated.')
#yr_built_input = input('Please enter the year the house was built.')

pred_df = pd.DataFrame({'bedrooms':bedroom_input,
			            'bathrooms':bathroom_input,
			            'condition':condition_input,
			            'lat':lat_input,
                        'floors':floors_input,
                        'long':long_input,
                        },index=[0])

preds = list(lm.predict(pred_df))
print('The featured estimated that the house rental price is',preds,'$')