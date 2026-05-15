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
sns.set_theme(style="whitegrid")

# Dados
st.subheader("Visualização dos Dados")
st.dataframe(df.head())

# Estatísticas
st.subheader("Estatísticas")
st.write(df.describe())

# Filtro interativo
species_filter = st.multiselect(
    "Filtrar espécies:",
    options=df['species'].unique(),
    default=df['species'].unique()
)

df_filtered = df[df['species'].isin(species_filter)]

# Scatter
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df_filtered,
    x='petal length (cm)',
    y='petal width (cm)',
    hue='species',
    s=100,
    ax=ax
)
# Controle manual dos limites
ax.set_xlim(0, 7)
ax.set_ylim(0, 3)

# Remove margens automáticas
ax.margins(x=0, y=0)

# Grid mais discreto
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

st.pyplot(fig)

# Histogramas
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

# Modelo
st.subheader("Modelo - Árvore de Decisão")

max_depth = st.slider("Max Depth", 1, 10, 3)

X = df.drop(columns=['target', 'species'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Cria o modelo
model = DecisionTreeClassifier(
    max_depth=max_depth,
    random_state=42
)
# Treina
model.fit(X_train, y_train)

# Predição
y_pred = model.predict(X_test)

# Acurácia
acc = accuracy_score(y_test, y_pred)
st.write(f"Acurácia: {acc:.4f}")

# Importância das variáveis
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

st.subheader("Importância das Features")
st.bar_chart(importance_df.set_index("Feature"))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
st.pyplot(fig)