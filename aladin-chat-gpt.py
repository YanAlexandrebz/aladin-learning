import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import requests
import openai

# Defina sua chave de API do OpenAI como uma variável de ambiente ou insira-a diretamente aqui
openai.api_key = 'sk-proj-St8h3W3zE5Qqquu5oUEQT3BlbkFJpS7cY8n5C3oFCPGTgr6K'

# Função para coletar dados de mudanças do jQuery
def get_jquery_changes():
    owner = "jquery"
    repo = "jquery"
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    response = requests.get(url)
    if response.status_code == 200:
        releases = response.json()
        changes = [release.get('body', '') for release in releases]
        return changes
    else:
        print("❌ Falha ao obter as mudanças do jQuery:", response.status_code)
        return None

# Função para vetorizar os dados de texto usando TF-IDF
def vectorize_text(data):
    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        vectorized_data = vectorizer.fit_transform(data)
        return vectorizer, vectorized_data
    except Exception as e:
        print("❌ Erro ao vetorizar os dados:", e)
        return None, None

# Função para interagir com o ChatGPT
def interact_with_chatgpt(prompt):
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      temperature=0.7,
      max_tokens=150
    )
    return response.choices[0].text.strip()

# Coletar dados de mudanças do jQuery
changes = get_jquery_changes()

if changes:
    print("✅ Dados do jQuery obtidos com sucesso!")
    
    # Carregar modelo treinado
    try:
        data = pd.read_csv('changes-conditions.csv')
        X_train, y_train = data['change_description'], data['relevant']
        vectorizer, X_train_vectorized = vectorize_text(X_train)
        
        if vectorizer is not None and X_train_vectorized is not None:
            model = SVC(kernel='linear')
            model.fit(X_train_vectorized, y_train)
            print("✅ Modelo treinado carregado com sucesso!")
            
            # Suponha que você tenha uma nova release do jQuery e queira verificar sua compatibilidade com a versão do seu projeto
            new_release_description = interact_with_chatgpt("Qual é a descrição da nova release do jQuery?")
            new_release_vectorized = vectorizer.transform([new_release_description])
            
            # Prever a compatibilidade da nova release com o seu projeto
            compatibility_prediction = model.predict(new_release_vectorized)
            
            # Exibir o resultado da previsão de compatibilidade
            print("\n🔮 Previsão de Compatibilidade:")
            if compatibility_prediction == 1:
                print("ℹ️ A nova release é considerada compatível com a versão do seu projeto.")
            else:
                print("⚠️ A nova release não é considerada compatível com a versão do seu projeto.")
        else:
            print("❌ Falha ao carregar o modelo treinado.")
    except Exception as e:
        print("❌ Ocorreu um erro ao carregar o modelo treinado:", e)
else:
    print("❌ Falha ao obter as mudanças do jQuery.")
