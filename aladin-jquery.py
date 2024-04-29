import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import requests

# Coletando dados de mudanças do jQuery
def get_jquery_changes():
    owner = "jquery"
    repo = "jquery"
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    response = requests.get(url)
    if response.status_code == 200:
        releases = response.json()
        changes = []
        for release in releases:
            if 'body' in release:
                changes.append(release['body'])
        return changes
    else:
        return None

# Vetorizando os dados de texto usando TF-IDF
def vectorize_text(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorized_data = vectorizer.fit_transform(data)
    return vectorizer, vectorized_data

# Gerando notas de release com base nas mudanças relevantes
def generate_release_notes(model, vectorizer, changes):
    relevant_changes = []
    for change in changes:
        vectorized_change = vectorizer.transform([change])
        prediction = model.predict(vectorized_change)[0]
        if prediction == 1:
            relevant_changes.append(change)
    return relevant_changes

# Coletando dados de mudanças do jQuery
changes = get_jquery_changes()

if changes:
    # Supondo que você tenha um modelo treinado anteriormente
    # Pode ser o modelo treinado em um conjunto de dados similar
    # Aqui vou criar um modelo de exemplo apenas para fins ilustrativos
    data = pd.read_csv('changes-conditions.csv')
    X_train, y_train = data['change_description'], data['relevant']
    vectorizer, X_train_vectorized = vectorize_text(X_train)
    model = SVC(kernel='linear')
    model.fit(X_train_vectorized, y_train)

    # Gerando notas de release com base nas mudanças relevantes
    relevant_changes = generate_release_notes(model, vectorizer, changes)

    # Exibindo as notas de release relevantes
    print("🚀📝🎉 Release Notes 🎉📝🚀")
    for change in relevant_changes:
        print("🔹 " + change)
else:
    print("Falha ao obter as mudanças do jQuery.")
