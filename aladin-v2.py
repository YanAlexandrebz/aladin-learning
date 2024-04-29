import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import requests

# Suponha que você tenha um arquivo CSV com dados de mudanças no repositório
# Este arquivo pode conter duas colunas: 'change_description' e 'relevant'
# 'change_description' contém o texto da mudança, 'relevant' indica se é relevante para o release notes (1) ou não (0)
data = pd.read_csv('changes-conditions.csv')

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['change_description'], data['relevant'], test_size=0.2)

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

def get_release_notes(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    response = requests.get(url)
    if response.status_code == 200:
        releases = response.json()
        release_notes = ""
        for release in releases:
            if 'body' in release:
                release_notes += release['body'] + "\n"
        return release_notes
    else:
        return None

# Após treinar e avaliar o modelo, você pode usá-lo para gerar release notes com base nas mudanças
def generate_release_notes(changes):
    relevant_changes = []
    for change in changes:
        vectorized_change = vectorizer.transform([change])
        prediction = model.predict(vectorized_change)[0]
        if prediction == 1:
            relevant_changes.append(change)
    return relevant_changes

owner = "python"
repo = "cpython"
changes = get_release_notes(owner,repo)
relevant_changes = generate_release_notes(changes)

print("******** Release Notes ********")
for change in relevant_changes:
    print("- " + change)
