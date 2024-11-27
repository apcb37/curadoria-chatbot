#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_Final.csv')

# Função de pré-processamento
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['pergunta'] = df['pergunta'].apply(preprocess_text)


# In[2]:


with open('Base_Conhecimento_Final.csv', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        print(f"Linha {i}: {line}")
        if i > 10:  # Para evitar imprimir todo o arquivo
            break


# In[4]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_Final.csv')

# Função de pré-processamento
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['pergunta'] = df['pergunta'].apply(preprocess_text)


# In[5]:


df = pd.read_csv('Base_Conhecimento_Final.csv', error_bad_lines=False, sep=',')


# In[6]:


df = pd.read_csv('Base_Conhecimento_Final.csv', sep=',', on_bad_lines='skip')


# In[7]:


import pandas as pd

# Carregar o arquivo, permitindo qualquer número de colunas
data = pd.read_csv('Base_Conhecimento_Final.csv', sep=',', header=None, on_bad_lines='skip')

# Encontrar o número máximo de colunas
max_columns = max(data.apply(lambda x: len(x), axis=1))

# Preencher linhas com menos colunas com valores nulos (ou um valor padrão, como 'NA')
data = data.apply(lambda x: x.tolist() + [None] * (max_columns - len(x)), axis=1)

# Converter para DataFrame novamente
data = pd.DataFrame(data.tolist())

# Renomear as colunas (opcional, se os nomes forem conhecidos)
data.columns = [f"Coluna_{i+1}" for i in range(max_columns)]

# Salvar o arquivo ajustado para evitar futuros erros
data.to_csv('Base_Conhecimento_Final_ajustado.csv', index=False)
print("Arquivo ajustado salvo com sucesso!")


# In[8]:


data = data.apply(lambda x: x.tolist() + ['NA'] * (max_columns - len(x)), axis=1)


# In[9]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_Final.csv')

# Função de pré-processamento
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['pergunta'] = df['pergunta'].apply(preprocess_text)


# In[10]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Baixar os recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Caminho do arquivo original (mantenha o mesmo nome)
file_path = 'Base_Conhecimento_Final.csv'

# 1. Carregar o arquivo, ignorando linhas problemáticas
try:
    raw_data = pd.read_csv(file_path, sep=',', header=None, on_bad_lines='skip', encoding='utf-8')
    print("Arquivo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")

# 2. Ajustar o número de colunas dinamicamente
max_columns = max(raw_data.apply(lambda x: len(x), axis=1))  # Encontra o maior número de colunas
raw_data = raw_data.apply(lambda x: x.tolist() + [None] * (max_columns - len(x)), axis=1)  # Completa as colunas
raw_data = pd.DataFrame(raw_data.tolist())  # Converte de volta para DataFrame

# 3. Renomear as colunas (opcional, se os nomes forem conhecidos)
column_names = ['Pergunta', 'Resposta', 'Categoria', 'Fonte', 'Data']  # Ajuste conforme necessário
raw_data.columns = column_names[:len(raw_data.columns)]  # Aplica os nomes disponíveis

# 4. Salvar o arquivo corrigido no mesmo caminho
raw_data.to_csv(file_path, index=False, encoding='utf-8')
print(f"Arquivo corrigido e salvo no mesmo caminho: {file_path}")

# 5. Recarregar o arquivo corrigido
try:
    df = pd.read_csv(file_path)
    print("Arquivo corrigido carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o arquivo corrigido: {e}")

# 6. Pré-processar os dados
# Função de pré-processamento
def preprocess_text(text):
    text = str(text)  # Garante que o texto seja uma string
    tokens = word_tokenize(text.lower())  # Tokeniza e converte para minúsculas
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]  # Remove stopwords
    return ' '.join(tokens)

# Aplicar o pré-processamento na coluna 'Pergunta'
if 'Pergunta' in df.columns:  # Verifica se a coluna 'Pergunta' existe
    df['Pergunta'] = df['Pergunta'].apply(preprocess_text)
    print("Pré-processamento concluído!")
else:
    print("Coluna 'Pergunta' não encontrada no arquivo corrigido!")

# Exibir as 5 primeiras linhas para verificação
print(df.head())


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

if 'Categoria' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Categoria', data=df, order=df['Categoria'].value_counts().index)
    plt.title('Distribuição de Perguntas por Categoria')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("A coluna 'Categoria' não foi encontrada no dataset.")


# In[12]:


print(df.isnull().sum())


# In[13]:


from sklearn.model_selection import train_test_split

X = df['Pergunta']  # Entrada (perguntas)
y = df['Categoria']  # Saída (categorias)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[18]:


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Baixar os recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Caminho do arquivo
file_path = 'Base_Conhecimento_Final.csv'

# Carregar o arquivo
df = pd.read_csv(file_path, encoding='utf-8')

# Função de pré-processamento
def preprocess_text(text):
    text = str(text)  # Garantir que o texto seja uma string
    tokens = word_tokenize(text.lower())  # Tokeniza e converte para minúsculas
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]  # Remove stopwords
    return ' '.join(tokens)

# Pré-processar perguntas e respostas
df['Pergunta'] = df['Pergunta'].apply(preprocess_text)
df['Resposta'] = df['Resposta'].apply(preprocess_text)

print(df.head())


# In[1]:


import pandas as pd

# Tente carregar o arquivo com múltiplos encodings, se necessário
file_path = df = pd.read_csv('Base_Conhecimento_Final.csv')  # Substitua pelo caminho real do arquivo

try:
    data = pd.read_csv(file_path, encoding="utf-8")
except:
    data = pd.read_csv(file_path, encoding="latin1")

# Adicionar uma nova coluna "Categoria" e categorizar as respostas
def categorize_response(response):
    if "tutorial" in str(response).lower():
        return "Tutorial"
    elif "proteção" in str(response).lower():
        return "Proteção Familiar"
    elif "sinistro" in str(response).lower():
        return "Sinistro"
    elif "renda" in str(response).lower():
        return "Renda Complementar"
    elif "benefício" in str(response).lower():
        return "Benefício"
    elif "indenizar" in str(response).lower():
        return "Indenização"
    else:
        return "Geral"

data["Categoria"] = data["Resposta"].apply(categorize_response)

# Salvar o resultado em um novo arquivo
output_file = "Base_Conhecimento_Categorizada.xlsx"
data.to_excel(output_file, index=False)

print(f"Arquivo categorizado salvo como {output_file}")


# In[2]:


import pandas as pd

# Substitua pelo caminho real do arquivo
file_path = 'Base_Conhecimento_Final.csv'

# Tente carregar o arquivo com múltiplos encodings, se necessário
try:
    data = pd.read_csv(file_path, encoding="utf-8")
except:
    data = pd.read_csv(file_path, encoding="latin1")

# Adicionar uma nova coluna "Categoria" e categorizar as respostas
def categorize_response(response):
    if "tutorial" in str(response).lower():
        return "Tutorial"
    elif "proteção" in str(response).lower():
        return "Proteção Familiar"
    elif "sinistro" in str(response).lower():
        
data["Categoria"] = data["Resposta"].apply(categorize_response)

# Salvar o resultado em um novo arquivo
output_file = "Base_Conhecimento_Categorizada.xlsx"
data.to_excel(output_file, index=False)

print(f"Arquivo categorizado salvo como {output_file}")


# In[4]:


import pandas as pd

# Substitua pelo caminho real do arquivo
file_path = 'Base_Conhecimento_Final.csv'

# Tente carregar o arquivo com múltiplos encodings, se necessário
try:
    data = pd.read_csv(file_path, encoding="utf-8")
except:
    data = pd.read_csv(file_path, encoding="latin1")

# Adicionar uma nova coluna "Categoria" e categorizar as respostas
def categorize_response(response):
    if "tutorial" in str(response).lower():
        return "Tutorial"
    elif "proteção" in str(response).lower():
        return "Proteção Familiar"
    elif "sinistro" in str(response).lower():


# In[5]:


import pandas as pd

# Substitua pelo caminho real do arquivo
file_path = 'Base_Conhecimento_Final.csv'

# Tente carregar o arquivo com múltiplos encodings, se necessário
try:
    data = pd.read_csv(file_path, encoding="utf-8")
except:
    data = pd.read_csv(file_path, encoding="latin1")

# Adicionar uma nova coluna "Categoria" e categorizar as respostas
def categorize_response(response):
    if "tutorial" in str(response).lower():
        return "Tutorial"
    elif "proteção" in str(response).lower():
        return "Proteção Familiar"
    elif "sinistro" in str(response).lower():
        


# In[6]:


import pandas as pd

# Substitua pelo caminho real do arquivo
file_path = 'Base_Conhecimento_Final.csv'

# Tente carregar o arquivo com múltiplos encodings, se necessário
try:
    data = pd.read_csv(file_path, encoding="utf-8")
except:
    data = pd.read_csv(file_path, encoding="latin1")

# Adicionar uma nova coluna "Categoria" e categorizar as respostas
def categorize_response(response):
    if "tutorial" in str(response).lower():
        return "Tutorial"
    elif "proteção" in str(response).lower():
        return "Proteção Familiar"
    elif "sinistro" in str(response).lower():
        return "Sinistro"
    elif "renda" in str(response).lower():
        return "Renda Complementar"
    elif "benefício" in str(response).lower():
        return "Benefício"
    elif "indenizar" in str(response).lower():
        return "Indenização"
    else:
        return "Geral"

data["Categoria"] = data["Resposta"].apply(categorize_response)

# Salvar o resultado em um novo arquivo
output_file = "Base_Conhecimento_Categorizada.xlsx"
data.to_excel(output_file, index=False)

print(f"Arquivo categorizado salvo como {output_file}")


# In[7]:


import pandas as pd

# Caminho do arquivo enviado
file_path = 'Base_Conhecimento_Final.csv'

# Tentativa de leitura do CSV
try:
    data = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
except UnicodeDecodeError:
    try:
        data = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        data = None
        error_message = str(e)

# Exibir informações básicas ou erro
if data is not None:
    display(data.head())  # Mostrar as primeiras linhas para ver o conteúdo
else:
    print(f"Erro ao carregar o arquivo: {error_message}")


# In[8]:


import pandas as pd

file_path = 'Base_Conhecimento_Corrigida.csv'  # Atualize com o caminho correto
data = pd.read_csv(file_path, encoding='utf-8')
data.head()


# In[9]:



# Importação das bibliotecas necessárias
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto
def preprocess_text(text):
    # Garantir que o texto seja uma string
    text = str(text)
    # Tokenizar e converter para minúsculas
    tokens = word_tokenize(text.lower())
    # Remover palavras que não sejam alfabéticas e stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicando o pré-processamento na coluna de perguntas
df['pergunta'] = df['pergunta'].apply(preprocess_text)

# Verificação inicial dos dados após o pré-processamento
print(df.head())

# Gráfico de distribuição de categorias
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()

# Vetorização utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['pergunta'])  # Matriz TF-IDF
y = df['categoria']  # Variável alvo

# Dividindo os dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Treinando o modelo de Rede Neural
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Avaliação dos modelos
# SVM
y_pred_svm = svm_model.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Rede Neural
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))


# In[10]:


# Exibir os nomes das colunas
print("Colunas disponíveis no DataFrame:", df.columns)

# Padronizar os nomes das colunas
df.columns = df.columns.str.lower().str.strip()

# Verificar se a coluna 'pergunta' existe
if 'pergunta' in df.columns:
    # Aplicar o pré-processamento
    df['pergunta'] = df['pergunta'].apply(preprocess_text)
else:
    raise KeyError("A coluna 'pergunta' não foi encontrada no arquivo CSV. Verifique o nome correto.")


# In[11]:


# Função de pré-processamento de texto
def preprocess_text(text):
    # Garantir que o texto seja uma string
    text = str(text)
    # Tokenizar e converter para minúsculas
    tokens = word_tokenize(text.lower())
    # Remover palavras que não sejam alfabéticas e stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicando o pré-processamento na coluna 'Pergunta'
df['Pergunta'] = df['Pergunta'].apply(preprocess_text)

# Verificação inicial dos dados após o pré-processamento
print(df.head())

# Gráfico de distribuição de categorias (ajuste se necessário)
if 'categoria' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
    plt.title('Distribuição de Perguntas por Categoria')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("A coluna 'categoria' não foi encontrada no DataFrame. Certifique-se de que ela existe.")

# Vetorização utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])  # Matriz TF-IDF

# Verificar se a coluna 'categoria' está presente para ser a variável alvo
if 'categoria' in df.columns:
    y = df['categoria']  # Variável alvo
else:
    raise KeyError("A coluna 'categoria' não foi encontrada no arquivo. Verifique o arquivo corrigido.")

# Dividindo os dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Treinando o modelo de Rede Neural
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Avaliação dos modelos
# SVM
y_pred_svm = svm_model.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Rede Neural
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))


# In[12]:


print(df.columns)


# In[13]:


# Remover espaços extras dos nomes das colunas
df.columns = df.columns.str.strip()


# In[14]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_.csv')

# Função de pré-processamento
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['pergunta'] = df['pergunta'].apply(preprocess_text)


# In[15]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['pergunta'] = df['pergunta'].apply(preprocess_text)


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Vetorização das perguntas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['pergunta'])
y = df['tem_resposta']  # Rótulo binário

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = svm_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[17]:


df['Pergunta'] = df['Pergunta'].apply(preprocess_text)


# In[18]:


# Importação das bibliotecas necessárias
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto
def preprocess_text(text):
    # Verificar se o texto é nulo e substituir por uma string vazia
    if pd.isnull(text):
        return ''
    # Garantir que o texto seja uma string
    text = str(text)
    # Tokenizar e converter para minúsculas
    tokens = word_tokenize(text.lower())
    # Remover palavras que não sejam alfabéticas e stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Tratar valores nulos na coluna 'Pergunta' antes de aplicar o preprocessamento
if 'Pergunta' in df.columns:
    df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)
else:
    raise KeyError("A coluna 'Pergunta' não foi encontrada no DataFrame.")

# Verificação inicial dos dados após o pré-processamento
print("Exemplo dos dados pré-processados:")
print(df.head())

# Gráfico de distribuição de categorias (se a coluna 'categoria' existir)
if 'categoria' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
    plt.title('Distribuição de Perguntas por Categoria')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("A coluna 'categoria' não foi encontrada no DataFrame. Certifique-se de que ela existe.")

# Vetorização utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])  # Matriz TF-IDF

# Verificar se a coluna 'categoria' está presente para ser a variável alvo
if 'categoria' in df.columns:
    y = df['categoria']  # Variável alvo
else:
    raise KeyError("A coluna 'categoria' não foi encontrada no arquivo. Verifique o arquivo corrigido.")

# Dividindo os dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Treinando o modelo de Rede Neural
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Avaliação dos modelos
# SVM
y_pred_svm = svm_model.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Rede Neural
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))


# In[19]:


print(df.columns)


# In[20]:


# Importação das bibliotecas necessárias
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto
def preprocess_text(text):
    # Verificar se o texto é nulo e substituir por uma string vazia
    if pd.isnull(text):
        return ''
    # Garantir que o texto seja uma string
    text = str(text)
    # Tokenizar e converter para minúsculas
    tokens = word_tokenize(text.lower())
    # Remover palavras que não sejam alfabéticas e stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Tratar valores nulos na coluna 'Pergunta' antes de aplicar o preprocessamento
if 'Pergunta' in df.columns:
    df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)
else:
    raise KeyError("A coluna 'Pergunta' não foi encontrada no DataFrame.")

# Função para categorizar automaticamente as perguntas
def categorize_question(question):
    question = question.lower()  # Converte a pergunta para minúsculas
    if "tutorial" in question:
        return "Tutorial"
    elif "proteção" in question:
        return "Proteção Familiar"
    elif "sinistro" in question:
        return "Sinistro"
    elif "renda" in question:
        return "Renda Complementar"
    elif "benefício" in question:
        return "Benefício"
    elif "indenizar" in question:
        return "Indenização"
    else:
        return "Geral"

# Aplicando a categorização automática
df['categoria'] = df['Pergunta'].apply(categorize_question)

# Verificação inicial dos dados após o pré-processamento
print("Exemplo dos dados pré-processados e categorizados:")
print(df.head())

# Gráfico de distribuição de categorias
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()

# Vetorização utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])  # Matriz TF-IDF

# A variável alvo é a coluna 'categoria', agora que ela foi criada
y = df['categoria']

# Dividindo os dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Treinando o modelo de Rede Neural
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Avaliação dos modelos
# SVM
y_pred_svm = svm_model.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Rede Neural
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))


# In[21]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto
def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['Pergunta'] = df['Pergunta'].apply(preprocess_text)

# Vetorização usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])
y = df['categoria']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balanceamento das classes com SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Treinamento do modelo SVM com ajuste de hiperparâmetros (Grid Search)
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid_search = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
svm_grid_search.fit(X_train_res, y_train_res)

# Treinamento do modelo de Rede Neural
param_grid_nn = {'hidden_layer_sizes': [(100,), (50, 50)], 'max_iter': [200, 500]}
nn_grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid_nn, cv=5)
nn_grid_search.fit(X_train_res, y_train_res)

# Avaliação do modelo SVM
y_pred_svm = svm_grid_search.best_estimator_.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Avaliação do modelo de Rede Neural
y_pred_nn = nn_grid_search.best_estimator_.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))

# Gráfico de distribuição de categorias (opcional)
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()


# In[22]:


pip install imbalanced-learn


# In[23]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV

# Carregar o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto
def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Aplicar o pré-processamento
df['Pergunta'] = df['Pergunta'].apply(preprocess_text)

# Vetorização usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])
y = df['categoria']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balanceamento das classes com SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Treinamento do modelo SVM com ajuste de hiperparâmetros (Grid Search)
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid_search = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
svm_grid_search.fit(X_train_res, y_train_res)

# Treinamento do modelo de Rede Neural
param_grid_nn = {'hidden_layer_sizes': [(100,), (50, 50)], 'max_iter': [200, 500]}
nn_grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid_nn, cv=5)
nn_grid_search.fit(X_train_res, y_train_res)

# Avaliação do modelo SVM
y_pred_svm = svm_grid_search.best_estimator_.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Avaliação do modelo de Rede Neural
y_pred_nn = nn_grid_search.best_estimator_.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))

# Gráfico de distribuição de categorias (opcional)
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Garanta que as colunas estejam corretamente definidas
df.columns = df.columns.str.strip()

# Verifique se a coluna 'categoria' está presente
if 'categoria' in df.columns:
    X = df['Pergunta']
    y = df['categoria']

    # Vectorização de texto
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Aplica o SMOTE para balanceamento
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Treinamento do modelo SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_res, y_train_res)

    # Previsões
    y_pred = svm_model.predict(X_test)

    # Avaliação do modelo
    print("Avaliação do Modelo SVM:")
    print(classification_report(y_test, y_pred))
else:
    print("A coluna 'categoria' não foi encontrada no DataFrame.")


# In[26]:


# Importação das bibliotecas necessárias
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Carregando o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto melhorada
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text).lower()
    tokens = word_tokenize(text)  # Tokenização
    # Remover palavras que não sejam alfabéticas e stopwords
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = stopwords.words('portuguese')
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Tratar valores nulos e aplicar o pré-processamento
if 'Pergunta' in df.columns:
    df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)
else:
    raise KeyError("A coluna 'Pergunta' não foi encontrada no DataFrame.")

# Melhorando a categorização automática
def categorize_question(question):
    question = question.lower()
    # Implementando mais palavras-chave e retornando categorias mais específicas
    if "tutorial" in question:
        return "Tutorial"
    elif "proteção" in question:
        return "Proteção Familiar"
    elif "sinistro" in question or "acidente" in question:
        return "Sinistro"
    elif "renda" in question or "complementar" in question:
        return "Renda Complementar"
    elif "benefício" in question or "auxílio" in question:
        return "Benefício"
    elif "indenizar" in question or "indenização" in question:
        return "Indenização"
    else:
        return "Geral"

# Aplicando a categorização automática
df['categoria'] = df['Pergunta'].apply(categorize_question)

# Verificação inicial dos dados após o pré-processamento
print("Exemplo dos dados pré-processados e categorizados:")
print(df.head())

# Gráfico de distribuição de categorias
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()


# In[27]:


# Importação das bibliotecas necessárias
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto
def preprocess_text(text):
    # Verificar se o texto é nulo e substituir por uma string vazia
    if pd.isnull(text):
        return ''
    # Garantir que o texto seja uma string
    text = str(text)
    # Tokenizar e converter para minúsculas
    tokens = word_tokenize(text.lower())
    # Remover palavras que não sejam alfabéticas e stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

# Tratar valores nulos na coluna 'Pergunta' antes de aplicar o preprocessamento
if 'Pergunta' in df.columns:
    df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)
else:
    raise KeyError("A coluna 'Pergunta' não foi encontrada no DataFrame.")

# Função para categorizar automaticamente as perguntas
def categorize_question(question):
    question = question.lower()  # Converte a pergunta para minúsculas
    if "tutorial" in question:
        return "Tutorial"
    elif "proteção" in question:
        return "Proteção Familiar"
    elif "sinistro" in question:
        return "Sinistro"
    elif "renda" in question:
        return "Renda Complementar"
    elif "benefício" in question:
        return "Benefício"
    elif "indenizar" in question:
        return "Indenização"
    else:
        return "Geral"

# Aplicando a categorização automática
df['categoria'] = df['Pergunta'].apply(categorize_question)

# Verificação inicial dos dados após o pré-processamento
print("Exemplo dos dados pré-processados e categorizados:")
print(df.head())

# Gráfico de distribuição de categorias
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()

# Vetorização utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])  # Matriz TF-IDF

# A variável alvo é a coluna 'categoria', agora que ela foi criada
y = df['categoria']

# Dividindo os dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Treinando o modelo de Rede Neural
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Avaliação dos modelos
# SVM
y_pred_svm = svm_model.predict(X_test)
print("Avaliação do Modelo SVM:")
print(classification_report(y_test, y_pred_svm))

# Rede Neural
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural:")
print(classification_report(y_test, y_pred_nn))


# In[28]:


# Instalando o imblearn, se necessário
# !pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Divisão dos dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicando SMOTE para balancear as classes no conjunto de treino
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f'Antes do SMOTE: {y_train.value_counts()}')
print(f'Depois do SMOTE: {y_train_smote.value_counts()}')


# In[29]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# Exibindo a distribuição das classes antes do SMOTE
print(f'Antes do SMOTE: {Counter(y_train)}')

# Verificando o número de amostras na classe minoritária
n_neighbors = min(len(set(y_train)), 6)  # Número de vizinhos não pode ser maior que o número de amostras

# Aplicando SMOTE para balancear as classes
smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Exibindo a distribuição das classes após o SMOTE
print(f'Depois do SMOTE: {Counter(y_train_smote)}')


# In[30]:


# Verificando o número de amostras por classe
class_counts = y_train.value_counts()
print("Distribuição das classes:", class_counts)

# Garantir que todas as classes tenham pelo menos 2 amostras
if any(class_counts < 2):
    print("Algumas classes possuem apenas 1 amostra. Considerando ajustes no dataset ou aplicação de técnicas alternativas.")


# In[31]:


from collections import Counter

# Verificando a distribuição das classes antes do SMOTE
print(f'Distribuição das classes no conjunto de treino: {Counter(y_train)}')


# In[32]:


from imblearn.over_sampling import SMOTE

# Ajustando o número de vizinhos com base no número de amostras
n_neighbors = min(len(set(y_train)), 6)  # Garantir que n_neighbors não seja maior que o número de amostras

# Aplicando SMOTE para balancear as classes
smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verificando o balanceamento das classes após o SMOTE
from collections import Counter
print(f'Depois do SMOTE: {Counter(y_train_smote)}')


# In[33]:


# Verificar se todas as categorias estão presentes no conjunto de treinamento
for category in set(y):
    if category not in y_train.values:
        print(f"A categoria '{category}' não está representada no conjunto de treinamento.")


# In[34]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Verificando a distribuição das classes antes do undersampling
print(f'Distribuição antes do undersampling: {Counter(y_train)}')

# Aplicando undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Verificando a distribuição das classes após o undersampling
print(f'Distribuição após o undersampling: {Counter(y_train_under)}')


# In[35]:


from sklearn.svm import SVC

# Modelo SVM com pesos ajustados automaticamente
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)

# Treinando o modelo SVM
svm_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred_svm = svm_model.predict(X_test)
from sklearn.metrics import classification_report
print("Avaliação do Modelo SVM com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_svm))


# In[36]:


from sklearn.neural_network import MLPClassifier

# Modelo de Rede Neural (MLP) com pesos ajustados
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, class_weight='balanced')

# Treinando o modelo de Rede Neural
nn_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_nn))


# In[37]:


import numpy as np

# Identificando os índices das amostras das classes minoritárias
minority_class_indices = np.where(y_train == 'Proteção Familiar')[0]  # Exemplo para a classe minoritária

# Replicando as amostras para aumentar sua quantidade
X_train_minority_oversampled = np.concatenate([X_train[minority_class_indices], X_train[minority_class_indices]])
y_train_minority_oversampled = np.concatenate([y_train[minority_class_indices], y_train[minority_class_indices]])

# Agora o conjunto de treino estará balanceado entre as classes
X_train_balanced = np.concatenate([X_train, X_train_minority_oversampled])
y_train_balanced = np.concatenate([y_train, y_train_minority_oversampled])

print(f"Distribuição balanceada das classes: {Counter(y_train_balanced)}")


# In[38]:


from sklearn.svm import SVC

# Modelo SVM com pesos ajustados automaticamente para balanceamento das classes
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)

# Treinando o modelo SVM com dados balanceados
svm_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred_svm = svm_model.predict(X_test)
from sklearn.metrics import classification_report
print("Avaliação do Modelo SVM com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_svm))


# In[39]:


pip install --upgrade scikit-learn


# In[40]:


from sklearn.neural_network import MLPClassifier

# Modelo de Rede Neural (MLP) com pesos ajustados
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, class_weight='balanced')

# Treinando o modelo de Rede Neural
nn_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred_nn = nn_model.predict(X_test)
print("Avaliação do Modelo de Rede Neural com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_nn))


# In[41]:


from sklearn.tree import DecisionTreeClassifier

# Modelo Árvore de Decisão com pesos ajustados
tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

# Treinando o modelo
tree_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred_tree = tree_model.predict(X_test)
print("Avaliação do Modelo Árvore de Decisão com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_tree))


# In[42]:


from sklearn.ensemble import RandomForestClassifier

# Modelo Random Forest com pesos ajustados
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Treinando o modelo
rf_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred_rf = rf_model.predict(X_test)
print("Avaliação do Modelo Random Forest com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_rf))


# In[43]:


import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Carregamento do dataset (ajuste conforme o local do seu arquivo)
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Pré-processamento de texto
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)

# Categorização automática (ajuste conforme suas categorias)
def categorize_question(question):
    question = question.lower()
    if "tutorial" in question:
        return "Tutorial"
    elif "proteção" in question:
        return "Proteção Familiar"
    elif "sinistro" in question:
        return "Sinistro"
    elif "renda" in question:
        return "Renda Complementar"
    elif "benefício" in question:
        return "Benefício"
    else:
        return "Geral"

df['categoria'] = df['Pergunta'].apply(categorize_question)

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])
y = df['categoria']

# Divisão em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificando o balanceamento das classes antes de qualquer técnica de balanceamento
print(f'Distribuição das classes no conjunto de treino antes do balanceamento: {Counter(y_train)}')

# Undersampling ou Oversampling manual (dependendo do caso)
# Exemplo de Oversampling manual:
# Vamos replicar a classe minoritária para balancear com as classes majoritárias
minority_class_indices = y_train[y_train == 'Proteção Familiar'].index  # Exemplo para classe minoritária

# Replicando as amostras para aumentar sua quantidade
X_train_minority_oversampled = X_train[minority_class_indices]
y_train_minority_oversampled = y_train[minority_class_indices]

# Agora o conjunto de treino estará balanceado entre as classes
X_train_balanced = X_train.append(X_train_minority_oversampled)
y_train_balanced = y_train.append(y_train_minority_oversampled)

# Verificando a distribuição das classes após o balanceamento
print(f'Distribuição das classes no conjunto de treino após o balanceamento: {Counter(y_train_balanced)}')

# Modelos com class_weight='balanced' para lidar com desbalanceamento de classes

# Árvore de Decisão com pesos ajustados
tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
tree_model.fit(X_train_balanced, y_train_balanced)
y_pred_tree = tree_model.predict(X_test)

print("Avaliação do Modelo Árvore de Decisão com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_tree))

# Random Forest com pesos ajustados
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf_model.predict(X_test)

print("Avaliação do Modelo Random Forest com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_rf))

# Verificando a distribuição final das classes no conjunto de teste para garantir que todas as categorias estão presentes
print(f'Distribuição das classes no conjunto de teste: {Counter(y_test)}')


# In[44]:


import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.sparse import vstack

# Carregamento do dataset (ajuste conforme o local do seu arquivo)
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Pré-processamento de texto
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('portuguese')]
    return ' '.join(tokens)

df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)

# Categorização automática (ajuste conforme suas categorias)
def categorize_question(question):
    question = question.lower()
    if "tutorial" in question:
        return "Tutorial"
    elif "proteção" in question:
        return "Proteção Familiar"
    elif "sinistro" in question:
        return "Sinistro"
    elif "renda" in question:
        return "Renda Complementar"
    elif "benefício" in question:
        return "Benefício"
    else:
        return "Geral"

df['categoria'] = df['Pergunta'].apply(categorize_question)

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])
y = df['categoria']

# Divisão em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibindo a distribuição das classes no conjunto de treino antes do balanceamento
print(f'Distribuição das classes no conjunto de treino antes do balanceamento: {Counter(y_train)}')

# Oversampling manual (aumento das amostras da classe minoritária)
minority_class_indices = y_train[y_train == 'Proteção Familiar'].index  # Exemplo para a classe minoritária

# Replicando as amostras para aumentar sua quantidade
X_train_minority_oversampled = X_train[minority_class_indices]
y_train_minority_oversampled = y_train[minority_class_indices]

# Agora o conjunto de treino estará balanceado entre as classes
X_train_balanced = vstack([X_train, X_train_minority_oversampled])  # Empilhando as amostras
y_train_balanced = np.concatenate([y_train, y_train_minority_oversampled])  # Concatenando as etiquetas

# Verificando a distribuição das classes após o balanceamento
print(f'Distribuição das classes no conjunto de treino após o balanceamento: {Counter(y_train_balanced)}')

# Modelos com class_weight='balanced' para lidar com desbalanceamento de classes

# Árvore de Decisão com pesos ajustados
tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
tree_model.fit(X_train_balanced, y_train_balanced)
y_pred_tree = tree_model.predict(X_test)

print("Avaliação do Modelo Árvore de Decisão com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_tree))

# Random Forest com pesos ajustados
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf_model.predict(X_test)

print("Avaliação do Modelo Random Forest com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_rf))

# Verificando a distribuição final das classes no conjunto de teste para garantir que todas as categorias estão presentes
print(f'Distribuição das classes no conjunto de teste: {Counter(y_test)}')


# In[45]:


# Importação das bibliotecas necessárias
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import vstack

# Carregando o dataset
df = pd.read_csv('Base_Conhecimento_Corrigida.csv')

# Função de pré-processamento de texto melhorada
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text).lower()  # Convertendo para minúsculas
    tokens = word_tokenize(text)  # Tokenização
    tokens = [word for word in tokens if word.isalpha()]  # Remover palavras não alfabéticas
    stop_words = stopwords.words('portuguese')  # Carregar stopwords em português
    tokens = [word for word in tokens if word not in stop_words]  # Remover stopwords
    return ' '.join(tokens)

# Tratar valores nulos e aplicar o pré-processamento
if 'Pergunta' in df.columns:
    df['Pergunta'] = df['Pergunta'].fillna('').apply(preprocess_text)
else:
    raise KeyError("A coluna 'Pergunta' não foi encontrada no DataFrame.")

# Melhorando a categorização automática
def categorize_question(question):
    question = question.lower()
    if "tutorial" in question:
        return "Tutorial"
    elif "proteção" in question:
        return "Proteção Familiar"
    elif "sinistro" in question or "acidente" in question:
        return "Sinistro"
    elif "renda" in question or "complementar" in question:
        return "Renda Complementar"
    elif "benefício" in question or "auxílio" in question:
        return "Benefício"
    elif "indenizar" in question or "indenização" in question:
        return "Indenização"
    else:
        return "Geral"

# Aplicando a categorização automática
df['categoria'] = df['Pergunta'].apply(categorize_question)

# Verificação inicial dos dados após o pré-processamento
print("Exemplo dos dados pré-processados e categorizados:")
print(df.head())

# Gráfico de distribuição de categorias
plt.figure(figsize=(10, 6))
sns.countplot(x='categoria', data=df, order=df['categoria'].value_counts().index)
plt.title('Distribuição de Perguntas por Categoria')
plt.xticks(rotation=45)
plt.show()

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Pergunta'])
y = df['categoria']

# Divisão em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibindo a distribuição das classes no conjunto de treino antes do balanceamento
from collections import Counter
print(f'Distribuição das classes no conjunto de treino antes do balanceamento: {Counter(y_train)}')

# Oversampling manual (aumento das amostras da classe minoritária)
minority_class_indices = y_train[y_train == 'Proteção Familiar'].index  # Exemplo para a classe minoritária

# Replicando as amostras para aumentar sua quantidade
X_train_minority_oversampled = X_train[minority_class_indices]
y_train_minority_oversampled = y_train[minority_class_indices]

# Agora o conjunto de treino estará balanceado entre as classes
X_train_balanced = vstack([X_train, X_train_minority_oversampled])  # Empilhando as amostras
y_train_balanced = np.concatenate([y_train, y_train_minority_oversampled])  # Concatenando as etiquetas

# Verificando a distribuição das classes após o balanceamento
print(f'Distribuição das classes no conjunto de treino após o balanceamento: {Counter(y_train_balanced)}')

# Modelos com class_weight='balanced' para lidar com desbalanceamento de classes

# Árvore de Decisão com pesos ajustados
tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
tree_model.fit(X_train_balanced, y_train_balanced)
y_pred_tree = tree_model.predict(X_test)

print("Avaliação do Modelo Árvore de Decisão com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_tree))

# Random Forest com pesos ajustados
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf_model.predict(X_test)

print("Avaliação do Modelo Random Forest com Ajuste de Pesos:")
print(classification_report(y_test, y_pred_rf))

# Verificando a distribuição final das classes no conjunto de teste para garantir que todas as categorias estão presentes
print(f'Distribuição das classes no conjunto de teste: {Counter(y_test)}')


# In[ ]:




