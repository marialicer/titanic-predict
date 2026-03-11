#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
#%%
df = pd.read_csv("data/Titanic-Dataset.csv")
# %%
df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# %%
X = df[['Pclass', 'Sex', 'Age', 'SibSp',	'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']
# %%
X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.2, random_state=42
)
# %%
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

# %%
joblib.dump(model, "model/titanic_model.pkl")

print("Modelo treinado e salvo!")
#%%