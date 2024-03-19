## Import
import numpy as numpy
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pickle

url = r".\data\raw\clean_house.csv"
house = pd.read_csv(url, sep=",")

def tranform_label_encoder(df, clist):
    for column in clist:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    return df

house = house[["property_subtype", "price", "surface_of_good", "swimming_pool", "state_of_building"]]
house = tranform_label_encoder(house, ["state_of_building"])

X = house.drop(columns="property_subtype")
y = house.property_subtype
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)

with open('model_pickle_dt', 'wb') as f:
    mp = pickle.dump(clf_entropy, f)

print(y_pred_en)

print(f"Accuracy is: {accuracy_score(y_test, y_pred_en)*100}")

predicttf = [[240000.0, 100.0, 1.0, 4]]

cl = clf_entropy.predict(predicttf)
print(cl)