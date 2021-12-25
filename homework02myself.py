import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("C:\\Users\\16873\Desktop\\advertising.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
regressor = LinearRegression()
regressor.fit(X, Y)
#regressor.fit(X_train, Y_train)
#print(regressor.score(X_test,Y_test))
print(regressor.predict([[100,100,100],[200,100,50]]))
