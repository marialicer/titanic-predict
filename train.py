#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import plot_tree
#%%

df = pd.read_csv("data/Titanic-Dataset.csv")
# %%
df.head()
# %%
df.info()
# %%
df.describe()
# %%
df.isnull().sum()
# %%
df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
# %%
df['Age'].fillna(df['Age'].mean(), inplace=True)
# %%
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# %%
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
# %%
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# %%
df.head()
# %%
X = df[['Pclass', 'Sex', 'Age', 'SibSp',	'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']
# %%
X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.2, random_state=42
)
# %%
model = DecisionTreeClassifier(max_depth=4)
# %%
model.fit(X_train,y_train)
# %%
y_pred = model.predict(X_test)
# %%
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# %%
print(classification_report(y_test, y_pred))
# %%
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Survived","Survived"],
    filled=True
)

plt.show()
# %%
y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %%
model.fit(X_train, y_train)

importances = model.feature_importances_
# %%
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})
# %%
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(feature_importance['Feature'], feature_importance['Importance'])

plt.title("Feature Importance - Decision Tree")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.show()
# %%
