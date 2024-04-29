import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Carregamento dos dados de teste
data = pd.read_csv('test_data.csv')

# Divisão dos dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['description'], data['result'], test_size=0.2)

# Vetorização dos dados de texto usando TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Treinamento do modelo de classificação (Support Vector Machine)
model = SVC(kernel='linear')
model.fit(X_train_vectorized, y_train)

# Avaliação do modelo
predictions = model.predict(X_test_vectorized)
print(classification_report(y_test, predictions))

# Geração de release notes com base nas previsões do modelo
def generate_release_notes(test_descriptions, model):
    release_notes = ""
    for description in test_descriptions:
        vectorized_description = vectorizer.transform([description])
        prediction = model.predict(vectorized_description)[0]
        if prediction == 'passou':
            release_notes += f"Teste '{description}' passou.\n"
        else:
            release_notes += f"Teste '{description}' falhou.\n"
    return release_notes

# Exemplo de uso
test_descriptions = [
    "Teste de unidade para função de soma",
    "Teste de integração com banco de dados",
    "Teste de performance para endpoint de API"
]

release_notes = generate_release_notes(test_descriptions, model)
print("Release Notes:")
print(release_notes)
