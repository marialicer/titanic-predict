#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
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
model = DecisionTreeClassifier(
    max_depth=4)
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
scores = cross_val_score(model, X, y, cv=5)

print(scores.mean())
# %%
param_grid = {
    'max_depth': [3,5,7,10],
    'min_samples_split': [2,5,10]
}
# %%
grid = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5
)
# %%
grid.fit(X_train, y_train)
# %%
print(grid.best_params_)
# %%
best_model = grid.best_estimator_
# %%
y_pred = best_model.predict(X_test)
# %%
y_pred_old = model.predict(X_test)
y_prob_old = model.predict_proba(X_test)[:,1]
# %%
y_pred_new = best_model.predict(X_test)
y_prob_new = best_model.predict_proba(X_test)[:,1]
# %%
acc_old = accuracy_score(y_test, y_pred_old)
acc_new = accuracy_score(y_test, y_pred_new)
# %%
auc_old = roc_auc_score(y_test, y_prob_old)
auc_new = roc_auc_score(y_test, y_prob_new)
# %%
print("Modelo original")
print("Accuracy:", acc_old)
print("AUC:", auc_old)

print("\nModelo otimizado (GridSearch)")
print("Accuracy:", acc_new)
print("AUC:", auc_new)
# %%
fpr_old, tpr_old, _ = roc_curve(y_test, y_prob_old)
fpr_new, tpr_new, _ = roc_curve(y_test, y_prob_new)

plt.figure()

plt.plot(fpr_old, tpr_old, label="Modelo original")
plt.plot(fpr_new, tpr_new, label="Modelo GridSearch")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
# %%
## Testar novos modelos, nesse caso o de RandomForest
from sklearn.ensemble import RandomForestClassifier
# %%
## Criação do modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)
# %%
#Treinamento do modelo
rf_model.fit(X_train, y_train)
# %%
#Fazer previsões com Random Forest
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]
# %%
## Calcular métricas do modelo Random Forest
acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("Random Forest Accuracy:", acc_rf)
print("Random Forest AUC:", auc_rf)
# %%
## Ver importância das variáveis no Random Forest
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)
# %%
## Plotar a importância das variáveis no Random Forest
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest")
plt.show()
# %%
## Ver matriz de confusão
cm = confusion_matrix(y_test, y_pred_rf)

print(cm)
# %%
## Importando o modelo de regressão logística
from sklearn.linear_model import LogisticRegression
# %%
## Criando o modelo de Regressão logística
log_model = LogisticRegression(max_iter=1000)
# %%
## Treinando o modelo de Regressão logística
log_model.fit (X_train, y_train)
# %%
## Fazendo previsões com o modelo de Regressão logística
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:,1]
# %%
## Avaliando o modelo de regressão logística
acc_log = accuracy_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_prob_log)

print("Logistic Regression Accuracy:", acc_log)
print("Logistic Regression AUC:", auc_log)
# %%
## Comparando com os outros modelos

results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Decision Tree (GridSearch)",
        "Random Forest",
        "Logistic Regression"
    ],
    "Accuracy": [
        acc_old,
        acc_new,
        acc_rf,
        acc_log
    ],
    "AUC": [
        auc_old,
        auc_new,
        auc_rf,
        auc_log
    ]
})

print(results)
# %%
## Calculando e plotando a curva ROC dos modelos para comparação
fpr_old, tpr_old, _ = roc_curve(y_test, y_prob_old)
fpr_new, tpr_new, _ = roc_curve(y_test, y_prob_new)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
# %%
plt.figure(figsize=(8,6))

plt.plot(fpr_old, tpr_old, label="Decision Tree")
plt.plot(fpr_new, tpr_new, label="Decision Tree (GridSearch)")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot(fpr_log, tpr_log, label="Logistic Regression")

plt.plot([0,1], [0,1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.show()
# %%
## Fazer cross validation no melhor modelo = Random Forest

scores = cross_val_score(rf_model, X, y, cv=5)

print(scores)
print(scores.mean())
# %%
## Tentar melhorar o Random Forest com GridSearch
param_grid = {
    "n_estimators": [100,200,300],
    "max_depth": [3,5,7,None],
    "min_samples_leaf": [1,3,5]
}
# %%
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_rf.fit(X_train, y_train)
# %%
## Ver os melhores parâmetros
print(grid_rf.best_params_)
# %%
## Pegar o melhor modelo
best_rf = grid_rf.best_estimator_
# %%
## Fazer previsões
y_pred_rf_new = best_rf.predict(X_test)
y_prob_rf_new = best_rf.predict_proba(X_test)[:,1]
# %%
## Calcular métricas novamente
acc_rf_new = accuracy_score(y_test, y_pred_rf_new)
auc_rf_new = roc_auc_score(y_test, y_prob_rf_new)

print("Random Forest original:", acc_rf, auc_rf)
print("Random Forest otimizado:", acc_rf_new, auc_rf_new)
# %%
## Comparar todos os modelos
results = pd.DataFrame({
    "Model":[
        "Decision Tree",
        "Random Forest",
        "Random Forest (Tuned)",
        "Logistic Regression"
    ],
    "Accuracy":[
        acc_old,
        acc_rf,
        acc_rf_new,
        acc_log
    ],
    "AUC":[
        auc_old,
        auc_rf,
        auc_rf_new,
        auc_log
    ]
})

print(results)
# %%
## Após encontrar o melhor modelo
best_rf = grid_rf.best_estimator_
# %%
## Salvar melhor modelo
import joblib

joblib.dump(best_rf, "titanic_model.pkl")
# %%
## Carregar o modelo
model = joblib.load("titanic_model.pkl")

print(model)
# %%
## Fazer nova predição manual

data = pd.DataFrame({
    "Pclass":[3],
    "Sex":[0],
    "Age":[25],
    "SibSp":[0],
    "Parch":[0],
    "Fare":[7.25],
    "Embarked_Q":[0],
    "Embarked_S":[1]
})

prediction = model.predict(data)

print(prediction)
# %%
