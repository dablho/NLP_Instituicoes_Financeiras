import streamlit as st
import pandas as pd
import psycopg2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import unicodedata
from collections import Counter
import networkx as nx
import os

# Definir o diretório para armazenar os dados do NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Adicionar o diretório ao caminho de dados do NLTK
nltk.data.path.append(nltk_data_dir)
nltk.download('all', download_dir=nltk_data_dir)

st.title("Análise de Sentimentos dos Comentários Negativos de Instituições Financeiras Brasileiras!")

def get_data_from_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("host"),
            database=os.getenv("database"),
            user=os.getenv("user"),
            password=os.getenv("password")
        )
        query = "SELECT * FROM prova.tabela_tcc"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

def remove_accent(text):
    return ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))

def preprocess_text(text, remove_stopwords=True):
    text = remove_accent(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    words = word_tokenize(text)
    if remove_stopwords:
        stop_words = set(stopwords.words('portuguese'))
        words = [word for word in words if word not in stop_words]
        
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def create_ngram_graph(text, n=4, top_n=10):
    words = preprocess_text(text, remove_stopwords=True)
    ngrams = list(nltk.ngrams(words, n))
    ngram_freq = Counter(ngrams)
    most_common_ngrams = ngram_freq.most_common(top_n)
    
    G = nx.DiGraph()
    
    for ngram, freq in most_common_ngrams:
        for i in range(len(ngram) - 1):
            G.add_edge(ngram[i], ngram[i + 1], weight=freq)
    
    return G, most_common_ngrams

def plot_ngram_bar_chart(ngrams, title):
    ngrams, counts = zip(*ngrams)
    ngrams = [" ".join(ngram) for ngram in ngrams]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ngrams, counts, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel('Frequência')
    plt.gca().invert_yaxis()
    st.pyplot(fig)

data = get_data_from_db()

if data is not None:
    st.write('Aqui estão os dados extraídos do banco de dados:')
    st.write(data)

    num_words = st.selectbox('Selecione o número de palavras mais frequentes para exibir nos gráficos de barras:', [10, 20, 30])

    st.write('Nuvem de palavras dos comentários (sem stopwords):')
    text_with_stopwords = " ".join(comment for comment in data['comentario'])
    wordcloud_with_stopwords = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_with_stopwords)

    fig_with_stopwords, ax_with_stopwords = plt.subplots(figsize=(10, 6))
    ax_with_stopwords.imshow(wordcloud_with_stopwords, interpolation='bilinear')
    ax_with_stopwords.axis("off")
    st.pyplot(fig_with_stopwords)

    st.write(f'Gráfico de barras das {num_words} palavras mais frequentes (sem stopwords):')
    words_with_stopwords = preprocess_text(text_with_stopwords, remove_stopwords=False)
    word_freq_with_stopwords = Counter(words_with_stopwords)
    most_common_with_stopwords = word_freq_with_stopwords.most_common(num_words)
    words, counts = zip(*most_common_with_stopwords)
    
    fig_bar_with_stopwords, ax_bar_with_stopwords = plt.subplots(figsize=(10, 6))
    ax_bar_with_stopwords.bar(words, counts)
    ax_bar_with_stopwords.set_title(f'{num_words} palavras mais frequentes (com stopwords)')
    ax_bar_with_stopwords.set_xlabel('Palavras')
    ax_bar_with_stopwords.set_ylabel('Frequência')
    ax_bar_with_stopwords.tick_params(axis='x', rotation=45)
    st.pyplot(fig_bar_with_stopwords)

    data['comentario'] = data['comentario'].apply(lambda x: " ".join(preprocess_text(x, remove_stopwords=True)))

    st.write('Nuvem de palavras dos comentários (com stopwords):')
    text_without_stopwords = " ".join(comment for comment in data['comentario'])
    wordcloud_without_stopwords = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_without_stopwords)

    fig_without_stopwords, ax_without_stopwords = plt.subplots(figsize=(10, 6))
    ax_without_stopwords.imshow(wordcloud_without_stopwords, interpolation='bilinear')
    ax_without_stopwords.axis("off")
    st.pyplot(fig_without_stopwords)

    st.write(f'Gráfico de barras das {num_words} palavras mais frequentes (com stopwords):')
    words_without_stopwords = preprocess_text(text_without_stopwords, remove_stopwords=False)
    word_freq_without_stopwords = Counter(words_without_stopwords)
    most_common_without_stopwords = word_freq_without_stopwords.most_common(num_words)
    words, counts = zip(*most_common_without_stopwords)
    
    fig_bar_without_stopwords, ax_bar_without_stopwords = plt.subplots(figsize=(10, 6))
    ax_bar_without_stopwords.bar(words, counts)
    ax_bar_without_stopwords.set_title(f'{num_words} palavras mais frequentes (sem stopwords)')
    ax_bar_without_stopwords.set_xlabel('Palavras')
    ax_bar_without_stopwords.set_ylabel('Frequência')
    ax_bar_without_stopwords.tick_params(axis='x', rotation=45)
    st.pyplot(fig_bar_without_stopwords)

    st.write(f'Grafo dos tetragramas mais frequentes e suas conexões:')
    G, most_common_ngrams = create_ngram_graph(text_without_stopwords, n=4, top_n=10)

    pos = nx.spring_layout(G, k=3, seed=20, iterations=50)
    fig_graph, ax_graph = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="skyblue", alpha=0.5, ax=ax_graph)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, arrowstyle='-|>', arrowsize=20)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black", ax=ax_graph)

    plt.title('Grafo dos tetragramas mais frequentes e suas conexões')
    st.pyplot(fig_graph)

    _, most_common_20_ngrams = create_ngram_graph(text_without_stopwords, n=4, top_n=20)
    plot_ngram_bar_chart(most_common_20_ngrams, 'Gráfico de barras dos 20 tetragramas mais frequentes')
else:
    st.write('Não foi possível carregar os dados.')

# Parte 2
st.subheader("Parte 2")

# Lista de stopwords personalizadas para serem adicionadas às stopwords do NLTK
additional_stopwords = {"para", "un", "exemplo", "ser", "poder", "so"}

# Define as stopwords incluindo as adicionais
stop_words = set(stopwords.words('portuguese')).union(additional_stopwords)

# Processa cada comentário individualmente e junta as palavras em uma string para a nuvem de palavras
text_with_stopwords = " ".join(" ".join(preprocess_text(comment, remove_stopwords=True)) for comment in data['comentario'])

# Nuvem de palavras com stopwords (incluindo stopwords adicionais) e preprocessamento
st.write('Nuvem de palavras dos comentários (com stopwords):')
wordcloud_with_stopwords = WordCloud(stopwords=stop_words, max_font_size=50, max_words=100, background_color="white").generate(text_with_stopwords)

fig_with_stopwords, ax_with_stopwords = plt.subplots(figsize=(10, 6))
ax_with_stopwords.imshow(wordcloud_with_stopwords, interpolation='bilinear')
ax_with_stopwords.axis("off")
st.pyplot(fig_with_stopwords)

# Gráfico de barras para palavras mais frequentes com stopwords personalizadas
st.write(f'Gráfico de barras das {num_words} palavras mais frequentes (com stopwords personalizadas):')

# Pré-processa o texto para o gráfico de barras, removendo as stopwords personalizadas
words_with_stopwords = [
    word for comment in data['comentario']
    for word in preprocess_text(comment, remove_stopwords=True)
    if word not in stop_words
]

word_freq_with_stopwords = Counter(words_with_stopwords)
most_common_with_stopwords = word_freq_with_stopwords.most_common(num_words)
words, counts = zip(*most_common_with_stopwords)

fig_bar_with_stopwords, ax_bar_with_stopwords = plt.subplots(figsize=(10, 6))
ax_bar_with_stopwords.bar(words, counts)
ax_bar_with_stopwords.set_title(f'{num_words} palavras mais frequentes (com stopwords personalizadas)')
ax_bar_with_stopwords.set_xlabel('Palavras')
ax_bar_with_stopwords.set_ylabel('Frequência')
ax_bar_with_stopwords.tick_params(axis='x', rotation=45)
st.pyplot(fig_bar_with_stopwords)
