
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
from textblob import Blobber
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from collections import defaultdict

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
additional_stopwords = {"para", "un", "exemplo", "ser", "poder", "so", "ja"}

# Define as stopwords incluindo as adicionais
stop_words = set(stopwords.words('portuguese')).union(additional_stopwords)

# Processa cada comentário individualmente e junta as palavras em uma string para a nuvem de palavras
text_with_stopwords = " ".join(" ".join(preprocess_text(comment, remove_stopwords=True)) for comment in data['comentario'])

# Nuvem de palavras com stopwords (incluindo stopwords adicionais) e preprocessamento
st.write('Nuvem de palavras dos comentários (com stopwords personalizadas):')
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


# Pré-processar comentários
data['comentario_tratado'] = data['comentario'].apply(preprocess_text)


#######





def nltk_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Determina o sentimento baseado no score composto
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positivo'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negativo'
    else:
        sentiment = 'Neutro'
    
    return {
        'sentiment_label': sentiment,
        'compound_score': sentiment_scores['compound'],
        'negative_score': sentiment_scores['neg'],
        'neutral_score': sentiment_scores['neu'],
        'positive_score': sentiment_scores['pos']
    }

# Aplica a análise de sentimentos
data['nltk_sentiment'] = data['comentario'].apply(nltk_sentiment)

# Expande as colunas para visualização
data['sentiment_label'] = data['nltk_sentiment'].apply(lambda x: x['sentiment_label'])
data['compound_score'] = data['nltk_sentiment'].apply(lambda x: x['compound_score'])
data['negative_score'] = data['nltk_sentiment'].apply(lambda x: x['negative_score'])
data['neutral_score'] = data['nltk_sentiment'].apply(lambda x: x['neutral_score'])
data['positive_score'] = data['nltk_sentiment'].apply(lambda x: x['positive_score'])

# Exibir tabela com análise de sentimentos
st.write("Análise de Sentimentos dos Comentários:")
st.write(data[['comentario', 'sentiment_label', 'compound_score', 'negative_score', 'neutral_score', 'positive_score']])



# Contagem de sentimentos
sentiment_counts = data['sentiment_label'].value_counts()

# Criar gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
sentiment_counts.plot(kind='bar', ax=ax, color=['red', 'gray', 'green'])
ax.set_title("Distribuição de Sentimentos nos Comentários", fontsize=15)
ax.set_xlabel("Sentimento", fontsize=12)
ax.set_ylabel("Número de Comentários", fontsize=12)

# Adicionar valores no topo de cada barra
for i, v in enumerate(sentiment_counts):
    ax.text(i, v, str(v), ha='center', va='bottom')

# Mostrar o gráfico no Streamlit
st.pyplot(fig)



### 



def custom_sentiment_analyzer(text):
    # Dicionário de palavras negativas específicas do contexto bancário
    negative_words = {
        'horrivel', 'ruim', 'pessimo', 'péssimo', 'péssima', 'problema', 
        'erro', 'dificil', 'difícil', 'frustração', 'nao funciona', 'não funciona', 
        'sem solução', 'demora', 'burocracia', 'complicado', 'confuso', 'irritante', 
        'péssimo', 'impossível', 'travado', 'lento', 'bug', 'falha', 'problema',
        'golpe', 'fraude', 'enganoso', 'mentiroso', 'nao paga', 'não paga',
        'limite', 'restritivo', 'abusivo', 'caro', 'injusto', 'negado','absurdo', 'abuso', 'fraude', 'calote', 'estelionato', 
        'incompetente', 'negligente', 'desrespeito', 'enganoso', 
        'manipulação', 'arbitrariedade', 'desumano'
    }
    
    # Dicionário de palavras positivas específicas do contexto bancário
    positive_words = {
        'bom', 'otimo', 'ótimo', 'excelente', 'rapido', 'rápido', 'facil', 
        'fácil', 'pratico', 'prático', 'simples', 'util', 'útil', 'funcional', 
        'direto', 'objetivo', 'rapido', 'rápido', 'seguro', 'confiavel', 
        'confiável', 'transparente', 'agil', 'ágil', 'moderno', 'eficiente', 'dedicado', 'inovador', 'transparente', 
        'acessível', 'responsável', 'sustentável', 'inclusivo'
    }
    
    # Normalização do texto
    text_lower = text.lower()
    
    # Contagem de palavras negativas e positivas
    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    # Análise de sentimento do TextBlob
    blob_sentiment = TextBlob(text).sentiment.polarity
    
    # Pontuação final de sentimento
    if negative_count > positive_count:
        sentiment_score = -abs(blob_sentiment)
    elif positive_count > negative_count:
        sentiment_score = abs(blob_sentiment)
    else:
        sentiment_score = blob_sentiment
    
    # Classificação do sentimento
    if sentiment_score <= -0.5:
        sentiment = 'Muito Negativo'
    elif -0.5 < sentiment_score < 0:
        sentiment = 'Negativo'
    elif sentiment_score == 0:
        sentiment = 'Neutro'
    elif 0 < sentiment_score < 0.5:
        sentiment = 'Positivo'
    else:
        sentiment = 'Muito Positivo'
    
    return {
        'texto': text,
        'sentimento': sentiment,
        'score': sentiment_score,
        'palavras_negativas': negative_count,
        'palavras_positivas': positive_count
    }


######

#####


def extract_aspects(comment):
    """
    Extrai aspectos específicos de comentários de aplicativos bancários
    """
    # Dicionários de palavras-chave por aspecto
    aspects_keywords = {
        'App': [
            'app', 'aplicativo', 'tela', 'interface', 'navegação', 'layout', 
            'design', 'bugado', 'travando', 'lento', 'crashando', 'fecha', 
            'instabilidade', 'performance', 'carregando'
        ],
        'Crédito': [
            'credito', 'crédito', 'limite', 'aprovação', 'emprestimo', 
            'empréstimo', 'financiamento', 'score', 'negativado', 'liberação'
        ],
        'Atendimento': [
            'atendimento', 'suporte', 'chat', 'central', 'ajuda', 'duvida', 
            'dúvida', 'reclamação', 'problema', 'resolução', 'tratamento'
        ],
        'Transferência': [
            'transferencia', 'transferência', 'pix', 'ted', 'pagamento', 
            'saldo', 'extrato', 'comprovante', 'transação'
        ],
        'Taxas': [
            'taxa', 'tarifa', 'custo', 'cobrança', 'mensalidade', 'anuidade', 
            'juros', 'spread', 'administração', 'manutenção'
        ],
        'Segurança': [
            'segurança', 'fraude', 'golpe', 'risco', 'proteção', 'biometria', 
            'senha', 'bloqueio', 'autenticação', 'invasão'
        ],
        'Conta Digital': [
            'conta', 'digital', 'online', 'saldo', 'extrato', 'cartão', 
            'virtual', 'movimento', 'cadastro', 'abertura'
        ]
    }
    
    # Conversão do comentário para minúsculas
    comment_lower = comment.lower()
    
    # Identificação de aspectos
    identified_aspects = defaultdict(int)
    
    for aspect, keywords in aspects_keywords.items():
        for keyword in keywords:
            if keyword in comment_lower:
                identified_aspects[aspect] += 1
    
    # Se nenhum aspecto for identificado, classifica como 'Geral'
    if not identified_aspects:
        identified_aspects['Geral'] = 1
    
    return dict(identified_aspects)

def sentiment_by_aspect(comment):
    """
    Análise de sentimento por aspecto
    """
    # Dicionário de palavras negativas por aspecto
    negative_keywords = {
        'App': ['travando', 'bugado', 'lento', 'crashando', 'fecha', 'ruim'],
        'Crédito': ['negado', 'baixo', 'impossível', 'restritivo', 'péssimo'],
        'Atendimento': ['péssimo', 'horrível', 'demora', 'incompetente', 'ignoram'],
        'Transferência': ['problema', 'erro', 'não funciona', 'travada'],
        'Taxas': ['abusivo', 'caro', 'injusto', 'excessivo'],
        'Segurança': ['vulnerável', 'inseguro', 'risco', 'problema'],
        'Conta Digital': ['complicado', 'confuso', 'difícil']
    }
    
    # Dicionário de palavras positivas por aspecto
    positive_keywords = {
        'App': ['rápido', 'intuitivo', 'fácil', 'bom', 'excelente'],
        'Crédito': ['aprovado', 'rápido', 'bom', 'alto', 'satisfatório'],
        'Atendimento': ['ótimo', 'rápido', 'eficiente', 'solícito'],
        'Transferência': ['instantâneo', 'fácil', 'rápido', 'prático'],
        'Taxas': ['justo', 'razoável', 'baixo'],
        'Segurança': ['seguro', 'confiável', 'protegido'],
        'Conta Digital': ['simples', 'prático', 'direto']
    }
    
    comment_lower = comment.lower()
    sentiment_scores = {}
    
    for aspect in negative_keywords.keys():
        # Conta palavras negativas
        negative_count = sum(keyword in comment_lower for keyword in negative_keywords[aspect])
        
        # Conta palavras positivas
        positive_count = sum(keyword in comment_lower for keyword in positive_keywords[aspect])
        
        # Calcula sentimento
        if negative_count > positive_count:
            sentiment_scores[aspect] = -1  # Negativo
        elif positive_count > negative_count:
            sentiment_scores[aspect] = 1   # Positivo
        else:
            sentiment_scores[aspect] = 0   # Neutro
    
    return sentiment_scores

def aspect_analysis(data):
    st.subheader("Análise de Aspectos")
    
    # Extrair aspectos de cada comentário
    data['aspectos'] = data['comentario'].apply(extract_aspects)
    
    # Análise de sentimento por aspecto
    data['sentimento_por_aspecto'] = data['comentario'].apply(sentiment_by_aspect)
    
    # Contagem de aspectos
    all_aspects = defaultdict(int)
    for aspectos in data['aspectos']:
        for aspecto, count in aspectos.items():
            all_aspects[aspecto] += count
    
    # Visualização de aspectos mais mencionados
    st.write("Aspectos mais mencionados")
    fig_aspectos, ax_aspectos = plt.subplots(figsize=(10, 6))
    plt.bar(all_aspects.keys(), all_aspects.values())
    plt.title("Distribuição de Aspectos nos Comentários")
    plt.xlabel("Aspectos")
    plt.ylabel("Número de Menções")
    plt.xticks(rotation=45)
    st.pyplot(fig_aspectos)
    
    # Análise de sentimento por aspecto
    st.write("Sentimento por Aspecto")
    sentiment_by_aspect_df = defaultdict(list)
    for aspectos, sentimentos in zip(data['aspectos'], data['sentimento_por_aspecto']):
        for aspecto in aspectos.keys():
            if aspecto in sentimentos:
                sentiment_by_aspect_df[aspecto].append(sentimentos[aspecto])
    
    # Cálculo da média de sentimento por aspecto
    sentiment_means = {
        aspecto: sum(sentimentos) / len(sentimentos) 
        for aspecto, sentimentos in sentiment_by_aspect_df.items()
    }
    
    # Visualização do sentimento médio por aspecto
    fig_sentimento, ax_sentimento = plt.subplots(figsize=(10, 6))
    plt.bar(sentiment_means.keys(), sentiment_means.values())
    plt.title("Sentimento Médio por Aspecto")
    plt.xlabel("Aspectos")
    plt.ylabel("Sentimento Médio")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(rotation=45)
    st.pyplot(fig_sentimento)
    
    # Tabela de aspectos e sentimentos
    st.write("Detalhes de Aspectos e Sentimentos")
    aspect_sentiment_df = pd.DataFrame.from_dict(
        {
            'Aspecto': list(sentiment_means.keys()),
            'Sentimento Médio': list(sentiment_means.values()),
            'Total de Menções': [all_aspects[aspecto] for aspecto in sentiment_means.keys()]
        }
    )
    st.dataframe(aspect_sentiment_df)
    
    # Comentários detalhados por aspecto
    st.write("Comentários por Aspecto")
    aspect_comments = defaultdict(list)
    
    for comentario, aspectos in zip(data['comentario'], data['aspectos']):
        for aspecto in aspectos.keys():
            aspect_comments[aspecto].append(comentario)
    
    # Mostrar alguns comentários por aspecto
    for aspecto, comentarios in aspect_comments.items():
        st.write(f"\n**Comentários sobre {aspecto}:**")
        for comentario in comentarios[:5]:  # Mostra 5 comentários por aspecto
            st.write(f"- {comentario}")

# Chamada da função
aspect_analysis(data)
