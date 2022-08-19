import pandas as pd
import os
import numpy as np
from flask import Flask
import pickle

#os.chdir('D:\test')

sd=pd.read_excel('/Users/biswaranjantripathy/Desktop/Datascience/fyi/salary.xlsx')
sd.head() 
x=sd.drop(['salary'],axis=1)
y=sd['salary']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
test=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
test.fit(x_train,y_train)
pickle.dump(test,open('mod.pkl','wb'))
mod=pickle.load(open('mod.pkl','rb'))
print(mod.predict(x_test))
#pred=test.predict(x_test)
#fg=np.round(pred)
#print(int(pred))
#print(y_test)

#from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(fg,y_test))
#print(confusion_matrix(fg,y_test))
