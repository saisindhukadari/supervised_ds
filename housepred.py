import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data={
    'weight':[150,120,110,140,130,128,100,193,126],
    'size':[7.0,7.5,7.6,7.8,6.5,6.7,5.4,7.4,9.0],
    'fruit':["Apple","Apple","Apple","Apple","Orange","Apple","Orange","Banana","Banana"]
}
df=pd.DataFrame(data)
df['encode_fruit']=df['fruit'].map({"Apple":0,"Orange":1,"Banana":2})
x=df[['weight','size']]
y=df['encode_fruit']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
sample=[[180,8.1]]
predicition=model.predict(sample)
label_map={0:"Apple",1:"Orange",2:"Banana"}
print(label_map[predicition[0]])
