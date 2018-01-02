#Importing important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#Reading the data from the csv file and examining the head, info and its description.
customers = pd.read_csv('Ecommerce Customers')

customers.head()

customers.describe()

customers.info()

#Exploring the data and plotting graphs
#plotting a joint plot btw 'Time on Website' and 'Yearly Amount Spent'.
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


#plotting a joint plot btw 'Time on App' and 'Yearly Amount Spent'.
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


#plotting a joint plot btw 'Time on App' and 'Yearly Amount Spent' in the 'Hex' format.
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers,kind='hex')


#Creating a general pairplot.
sns.pairplot(customers)

#Creating a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


#Traing and testing the data using linear regression model.
customers.columns


#Initializing the variables.
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']


#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#Training the Model
from sklearn.linear_model import LinearRegression

im = LinearRegression()

im.fit(X_train,y_train)

predictions = im.predict(X_test)

#Creating a scatterplot of the real test values versus the predicted values.
plt.scatter(predictions,y_test)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


#Evaluating the Model
# Calculating the residual sum of squares and the explained variance score (R^2).
# Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE: {}'.format(metrics.mean_absolute_error(y_test,predictions)))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


#Ploting a histogram of the residuals and makeing sure it looks normally distributed.
sns.distplot((y_test-predictions), bins=50)

#Recreating Coefficients dataframe
coff_df = pd.DataFrame(im.coef_, X.columns,columns = ['Coeffecient'])
coff_df


