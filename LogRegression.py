import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data={
    'maths':[78,45,90,35,65,82,33,45,85,90],
    'physics':[80,40,65,31,75,38,69,47,69,32],
    'chemistry':[42,98,68,75,36,48,69,31,85,46],
    'Result':['pass','pass','pass','fail','pass','pass','fail','fail','pass','fail']
}
#convert data into data frame
df=pd.DataFrame(data)
df['Result']=df['Result'].map({'pass':1,'fail':0})
x=df[['maths','physics','chemistry']]
y=df['Result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))
new_stu=pd.DataFrame([[41,15,61]],columns=['maths','physics','chemistry'])
prediction=model.predict(new_stu)
print(prediction[0])
