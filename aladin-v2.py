import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

class ReleaseNotesGenerator:
    def __init__(self, previous_code, new_code):
        self.previous_code = previous_code
        self.new_code = new_code

    def find_code_changes(self):
        """
        Encontra as diferenças entre o código anterior e o novo código.
        """
        previous_lines = self.previous_code.splitlines()
        new_lines = self.new_code.splitlines()

        matcher = SequenceMatcher(None, previous_lines, new_lines)
        diffs = matcher.get_opcodes()

        changes = []
        for tag, i1, i2, j1, j2 in diffs:
            if tag == 'replace':
                changes.extend(new_lines[j1:j2])
            elif tag == 'insert':
                changes.extend(new_lines[j1:j2])
            elif tag == 'delete':
                pass  # Ignorar linhas deletadas
        return changes

    def generate_release_notes(self):
        """
        Gera notas de lançamento com base nas mudanças no código.
        """
        changes = self.find_code_changes()

        # Se houverem mudanças, podemos usar um TF-IDF para extrair palavras-chave
        if changes:
            # Converter as mudanças para texto
            changes_text = ' '.join(changes)

            # Criar um DataFrame para armazenar os dados
            data = {'Changes': [changes_text]}
            df = pd.DataFrame(data)

            # Criar um vetorizador TF-IDF
            vectorizer = TfidfVectorizer()

            # Ajustar e transformar os dados
            tfidf_matrix = vectorizer.fit_transform(df['Changes'])

            # Calcular a similaridade de cosseno entre as mudanças e as palavras-chave pré-definidas
            similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Index da mudança mais similar
            similar_index = similarity.argsort()[0][-1]

            # Notas de lançamento baseadas na mudança mais similar
            release_notes = df['Changes'][similar_index]

            return release_notes
        else:
            return "Sem mudanças no código. Nenhuma nota de lançamento gerada."


          # Exemplo de uso
          previous_code = """
          def hello_world():
              print("Hello, world!")
          """
          
          new_code = """
          def meu_codigo():
              print("oh neymar")
          
          def hello_world():
              print("Hello, world!")
              meu_codigo()
              print("Goodbye, world!")
          """
          
          generator = ReleaseNotesGenerator(previous_code, new_code)
          release_notes = generator.generate_release_notes()
          print(release_notes)
