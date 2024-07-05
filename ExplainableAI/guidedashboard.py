from explainerdashboard.datasets import titanic_embarked, titanic_names
from explainerdashboard import ClassifierExplainer, ExplainerDashboard


X_train, y_train, X_test, y_test = titanic_embarked()
train_names, test_names = titanic_names()

print(train_names)
print(X_train)
print(y_train)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                                    cats=['Sex', 'Deck'],
                                    idxs=test_names, 
                                    labels=['Queenstown', 'Southampton', 'Cherbourg'],
                                    pos_label='Southampton')

ExplainerDashboard(explainer, mode='external').run()
