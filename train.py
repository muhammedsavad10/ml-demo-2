from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

x,y=load_iris(return_X_y=True)
model=RandomForestClassifier()
model.fit(x,y)

os.makedirs("model",exist_ok=True)

with open("model/model.pkl","wb") as f:
    pickle.dump(model,f)