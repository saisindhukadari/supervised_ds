import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("Housing.csv")
#print(df.head())
#print(df.columns)
x=df[['area','bedrooms']]
df['price']=df['price'].astype(float)
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train)
model=DecisionTreeRegressor()
model.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# print("Accuracy:",accuracy_score(y_pred,y_test))
new_house=[[3750,4]]
prediction=model.predict(new_house)
print(prediction)