import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Training completed successfully")