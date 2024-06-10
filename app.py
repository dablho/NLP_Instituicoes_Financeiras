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

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.title("Análise de Sentimentos dos Comentários Negativos de Instituições Financeiras Brasileiras!")

def get_data_from_db():
    try:
        conn = psycopg2.connect(
            host="dataiesb.iesbtech.com.br",
            database="2312120030_William",
            user="2312120030_William",
            password="2312120030_William"
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
    # Remove acentos
    text = remove_accent(text)
    # Converter para minúsculas
    text = text.lower()
    # Remove pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove números
    text = ''.join([i for i in text if not i.isdigit()])
    # Tokenização
    words = word_tokenize(text)
    if remove_stopwords:
        # Remover stopwords
        stop_words = set(stopwords.words('portuguese'))
        words = [word for word in words if word not in stop_words]
    # Lematização
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
    plt.gca().invert_yaxis()  # Inverter o eixo y para colocar a maior barra no topo
    st.pyplot(fig)

# Obter dados do banco de dados
data = get_data_from_db()

if data is not None:
    # Mostrar os dados no Streamlit
    st.write('Aqui estão os dados extraídos do banco de dados:')
    st.write(data)

    # Selecionar o número de palavras mais frequentes
    num_words = st.selectbox('Selecione o número de palavras mais frequentes para exibir nos gráficos de barras:', [10, 20, 30])

    # Nuvem de palavras com stopwords
    st.write('Nuvem de palavras dos comentários (com stopwords):')
    text_with_stopwords = " ".join(comment for comment in data['comentario'])
    wordcloud_with_stopwords = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_with_stopwords)

    fig_with_stopwords, ax_with_stopwords = plt.subplots(figsize=(10, 6))
    ax_with_stopwords.imshow(wordcloud_with_stopwords, interpolation='bilinear')
    ax_with_stopwords.axis("off")
    st.pyplot(fig_with_stopwords)

    # Gráfico de barras para palavras mais frequentes com stopwords
    st.write(f'Gráfico de barras das {num_words} palavras mais frequentes (com stopwords):')
    words_with_stopwords = preprocess_text(text_with_stopwords, remove_stopwords=False)
    word_freq_with_stopwords = Counter(words_with_stopwords)
    most_common_with_stopwords = word_freq_with_stopwords.most_common(num_words)
    words, counts = zip(*most_common_with_stopwords)
    
    fig_bar_with_stopwords, ax_bar_with_stopwords = plt.subplots(figsize=(10, 6))
    ax_bar_with_stopwords.bar(words, counts)
    ax_bar_with_stopwords.set_title(f'{num_words} palavras mais frequentes (com stopwords)')
    ax_bar_with_stopwords.set_xlabel('Palavras')
    ax_bar_with_stopwords.set_ylabel('Frequência')
    ax_bar_with_stopwords.tick_params(axis='x', rotation=45)  # Rotaciona os rótulos para 45 graus
    st.pyplot(fig_bar_with_stopwords)

    # Preprocessar os comentários para remover stopwords
    data['comentario'] = data['comentario'].apply(lambda x: " ".join(preprocess_text(x, remove_stopwords=True)))

    # Nuvem de palavras sem stopwords
    st.write('Nuvem de palavras dos comentários (sem stopwords):')
    text_without_stopwords = " ".join(comment for comment in data['comentario'])
    wordcloud_without_stopwords = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_without_stopwords)

    fig_without_stopwords, ax_without_stopwords = plt.subplots(figsize=(10, 6))
    ax_without_stopwords.imshow(wordcloud_without_stopwords, interpolation='bilinear')
    ax_without_stopwords.axis("off")
    st.pyplot(fig_without_stopwords)


    # Gráfico de barras para palavras mais frequentes sem stopwords
    st.write(f'Gráfico de barras das {num_words} palavras mais frequentes (sem stopwords):')
    words_without_stopwords = preprocess_text(text_without_stopwords, remove_stopwords=False)
    word_freq_without_stopwords = Counter(words_without_stopwords)
    most_common_without_stopwords = word_freq_without_stopwords.most_common(num_words)
    words, counts = zip(*most_common_without_stopwords)
    
    fig_bar_without_stopwords, ax_bar_without_stopwords = plt.subplots(figsize=(10, 6))
    ax_bar_without_stopwords.bar(words, counts)
    ax_bar_without_stopwords.set_title(f'{num_words} palavras mais frequentes (sem stopwords)')
    ax_bar_without_stopwords.set_xlabel('Palavras')
    ax_bar_without_stopwords.set_ylabel('Frequência')
    ax_bar_without_stopwords.tick_params(axis='x', rotation=45)  # Rotaciona os rótulos para 45 graus
    st.pyplot(fig_bar_without_stopwords)

    # Criar grafo de tetragramas
    st.write(f'Grafo dos tetragramas mais frequentes e suas conexões:')
    G, most_common_ngrams = create_ngram_graph(text_without_stopwords, n=4, top_n=10)  # Usar 10 tetragramas mais frequentes

    # Definir a seed para o layout do grafo
    pos = nx.spring_layout(G, k=3, seed=20, iterations=50)  # Ajuste o valor de k para controlar o espaçamento e aumente as iterações para melhor convergência

    fig_graph, ax_graph = plt.subplots(figsize=(12, 8))

    # Desenhar os nós com transparência
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="skyblue", alpha=0.5, ax=ax_graph)

    # Desenhar as arestas com setas
    nx.draw_networkx_edges(G, pos, ax=ax_graph, arrowstyle='-|>', arrowsize=20)

    # Desenhar os rótulos dos nós
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black", ax=ax_graph)

    plt.title('Grafo dos tetragramas mais frequentes e suas conexões')
    st.pyplot(fig_graph)

    # Obter os 20 tetragramas mais frequentes para o gráfico de barras
    _, most_common_20_ngrams = create_ngram_graph(text_without_stopwords, n=4, top_n=20)
    
    # Gráfico de barras dos 20 tetragramas mais frequentes
    plot_ngram_bar_chart(most_common_20_ngrams, 'Gráfico de barras dos 20 tetragramas mais frequentes')
else:
    st.write('Não foi possível carregar os dados.')
