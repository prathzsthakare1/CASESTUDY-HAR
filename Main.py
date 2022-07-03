import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('train.csv')
dtc = DecisionTreeClassifier()
svm = svm.SVC()
gbc = GradientBoostingClassifier(n_estimators=10)
logr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)

data = pd.read_csv('train.csv')

x = data.drop(['Activity','subject'], axis=1)
y = data['Activity'].astype(object)

le = LabelEncoder()
y = le.fit_transform(y)


scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)

logr.fit(x_train, y_train)
rfc.fit(x_train, y_train)
dtc.fit(x_train, y_train)
svm.fit(x_train, y_train)
gbc.fit(x_train, y_train)

ylogr_predict = logr.predict(x_test)
rfcy_predict = rfc.predict(x_test)
dtcy_predict = dtc.predict(x_test)
svmy_predict = svm.predict(x_test)
gbcy_predict = gbc.predict(x_test)

print('Logistic:', accuracy_score(y_test, ylogr_predict))
print('Random Forest:', accuracy_score(y_test, rfcy_predict))
print('Decision Tree:', accuracy_score(y_test, dtcy_predict))
print('Support Vector:', accuracy_score(y_test, svmy_predict))
print('Gradient Boosting:', accuracy_score(y_test,  gbcy_predict))

'''
Accuracy Scores:
Logistic: 0.9714396735962697
Random Forest: 0.9677482028366039
Decision Tree: 0.9209248105692637
Support Vector: 0.9638624441422188
Gradient Boosting: 0.9413250437147853
'''
