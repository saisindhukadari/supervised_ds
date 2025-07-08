import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

data={
    'message':['Win a free lottery ticket','hi,find your attachment file',
               'kudos,you won a car','meet you at 10am tomorrow','claim your free gift'
               ,'terroist planned bomb blast in mrecw','surrendar is very good boy'],
    'status':['spam','not spam','spam','not spam','spam','spam','spam']
}
#convert to dataframe
df=pd.DataFrame(data)
#map output variables into binary format
df['status']=df['status'].map({'spam':1,'not spam':0})
#declare input and output variable columns into x and y
x=df['message']
y=df['status']
vectorizer=CountVectorizer()
x_vectorizer=vectorizer.fit_transform(x)
#train and test the model
x_train,x_test,y_train,y_test=train_test_split(x_vectorizer,y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_pred,y_test))
sample_message=["free prize waiting for you"]
sample_vector=vectorizer.transform(sample_message)
prediction=model.predict(sample_vector)
print(prediction[0])
