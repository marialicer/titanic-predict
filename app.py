# %%
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("titanic_model.pkl")

st.title("Você sobreveviria ao Titanic? Responda e Descubra!")


classe = st.selectbox(
    "Classe do Passageiro",
    ["1ª Classe", "2ª Classe", "3ª Classe"]
)

sexo = st.selectbox(
    "Sexo",
    ["Masculino", "Feminino"]
)

idade = st.slider("Idade", 0, 80, 30)

sibsp = st.slider("Número de irmãos/cônjuge a bordo", 0, 5, 0)

parch = st.slider("Número de pais/filhos a bordo", 0, 5, 0)

fare = st.slider("Valor da passagem", 0, 500, 50)
# %%
pclass_map = {
    "1ª Classe": 1,
    "2ª Classe": 2,
    "3ª Classe": 3
}

sex_map = {
    "Masculino": 0,
    "Feminino": 1
}
# %%

if st.button("Prever sobrevivência"):

    data = pd.DataFrame({
        "Pclass":[pclass_map[classe]],
        "Sex":[sex_map[sexo]],
        "Age":[idade],
        "SibSp":[sibsp],
        "Parch":[parch],
        "Fare":[fare],
        "Embarked_Q":[0],
        "Embarked_S":[1]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("O passageiro provavelmente sobreviveria.")
    else:
        st.error("O passageiro provavelmente não sobreviveria.")
# %%
