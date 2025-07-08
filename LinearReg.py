import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
data={
    'hrs_studied':[1,2,3,4,5,6,7,8,9,10],
    'final_score':[20,35,45,50,65,70,75,80,85,95],
}
df=pd.DataFrame(data)
# prepare data
x=df[['hrs_studied']]
y=df['final_score']
# split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#model creation and call
model=LinearRegression()
model.fit(x_train,y_train)
#predict the model
y_pred=model.predict(x_test)
print("mean_squared_error:",mean_squared_error(y_test,y_pred))
print("R score",r2_score(y_test,y_pred))
new_hours=pd.DataFrame([[7.5]],columns=['hrs_studied'])
prediction_score=model.predict(new_hours)
print(prediction_score[0])