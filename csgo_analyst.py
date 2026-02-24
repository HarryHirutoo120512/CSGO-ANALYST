import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
data = pd.read_csv("csgo.csv")
x = data.drop(columns = ["wait_time_s","match_time_s","team_a_rounds","team_b_rounds","result","map","date"],axis=1)
y = data["result"].map({
    "Win" : 1,
    "Tie" : 1,
    "Lost" : 0,
})
enc = OneHotEncoder(handle_unknown='ignore')
map_encoded = enc.fit_transform(data[["map"]])
#encoder cột map thành one-hot
map_encoded_df = pd.DataFrame(map_encoded, columns=["map"])
#ghép vào data frame
data = pd.concat([data, map_encoded_df], axis=1)
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
parameters = {
    "n_estimators": [50, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [None,5,10],
}
cls = GridSearchCV(RandomForestClassifier(), param_grid=parameters,cv=3, scoring="precision", n_jobs = 1, verbose = 2)
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
print(cls.best_score_)
print(cls.best_params_)
print(classification_report(y_test, y_pred))