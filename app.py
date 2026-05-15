import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuração
st.set_page_config(page_title="Dashboard Iris", layout="wide")

# Carregamento
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

st.title("Dashboard - Iris Dataset")

# 🔹 Dados
st.subheader("Visualização dos Dados")
st.dataframe(df.head())

# 🔹 Estatísticas
st.subheader("Estatísticas")
st.write(df.describe())

# 🔹 Filtro interativo
species_filter = st.multiselect(
    "Filtrar espécies:",
    options=df['species'].unique(),
    default=df['species'].unique()
)

df_filtered = df[df['species'].isin(species_filter)]

# 🔹 Scatter
st.subheader("Relação entre Variáveis")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_filtered,
    x='petal length (cm)',
    y='petal width (cm)',
    hue='species',
    ax=ax
)
st.pyplot(fig)

# 🔹 Histogramas
st.subheader("Distribuição")

feature_choice = st.selectbox(
    "Selecione a variável:",
    [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    data=df_filtered,
    x=feature_choice,
    hue='species',
    multiple='dodge',
    kde=True,
    ax=ax
)

plt.tight_layout()

st.pyplot(fig)

# 🔹 Modelo
st.subheader("Modelo - Árvore de Decisão")

X = df.drop(columns=['target', 'species'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Acurácia: {acc:.4f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
st.pyplot(fig)