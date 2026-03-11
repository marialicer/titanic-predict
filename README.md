# Titanic Survival Prediction



Este projeto utiliza **Machine Learning** para prever se um passageiro sobreviveria ao desastre do Titanic com base em características como classe, sexo, idade e outras variáveis.



O modelo foi treinado utilizando o dataset clássico do Titanic e disponibilizado em uma interface interativa construída com **Streamlit**.



---



## Objetivo do Projeto



Construir e comparar diferentes modelos de classificação para prever a sobrevivência dos passageiros do Titanic.



Modelos testados:



- Decision Tree

- Random Forest

- Logistic Regression



O melhor desempenho foi obtido com **Random Forest otimizado com GridSearchCV**.



---



## Estrutura do Projeto



titanic-ml-project/



data/

Titanic-Dataset.csv



model/

titanic_model.pkl



notebooks/

titanic_analysis.ipynb



train_model.py

app.py

requirements.txt

README.md



---



## Tecnologias Utilizadas



- Python

- Pandas

- NumPy

- Scikit-learn

- Matplotlib

- Streamlit

- Joblib



---



## Como Rodar o Projeto



Instale as dependências:



pip install -r requirements.txt



Treine o modelo:



python train_model.py



Execute o app:



streamlit run app.py



Depois abra no navegador:



http://localhost:8501



---



## Como Funciona



O usuário insere as características do passageiro:



* Classe do Passageiro



* Sexo



* Idade



* Número de irmãos/cônjuges a bordo



* Número de pais/filhos a bordo



* Valor da passagem



O modelo então prevê:



* Sobreviveu



* Não sobreviveu



---



## Autor



Projeto desenvolvido para prática de Machine Learning e deploy de modelos com Streamlit.



---



## Dataset



Dataset utilizado:



Titanic Dataset (Kaggle)



https://www.kaggle.com/datasets/yasserh/titanic-dataset













