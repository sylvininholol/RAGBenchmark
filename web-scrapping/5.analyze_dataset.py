import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
import time

load_dotenv()

def categorizar_esporte_llm(titulos: list[str], llm_client: OpenAI) -> list[str]:
    """Classifica títulos de notícias pelo esporte específico usando um LLM."""

    instrucao_sistema = """
    Você é um classificador de notícias esportivas. Sua tarefa é ler o título fornecido
    e identificar o esporte específico sendo discutido.
    Responda APENAS com o nome do esporte (ex: Futebol, Basquete, Tênis, Automobilismo, Vôlei, Surfe, MMA, Boxe, eSports, Atletismo).
    Se o título for muito genérico sobre esportes ou não ficar claro qual o esporte principal,
    responda 'Esportes Gerais'. Se não parecer ser sobre esportes, responda 'Não Esportivo'.
    Seja o mais específico possível (ex: 'Fórmula 1' em vez de 'Automobilismo' se o título mencionar F1).
    """

    classificacoes = []
    total_titulos = len(titulos)
    
    for i, titulo in enumerate(titulos):
        if not isinstance(titulo, str) or not titulo.strip():
            print(f"Título {i+1}/{total_titulos}: Inválido ou vazio. Classificando como 'Outros'.")
            classificacoes.append('Outros')
            continue

        print(f"Classificando título {i+1}/{total_titulos}: '{titulo[:60]}...'")

        tentativas = 0
        max_tentativas = 3
        wait_time = 5
        
        while tentativas < max_tentativas:
            try:
                response = llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": instrucao_sistema},
                        {"role": "user", "content": f"Título: \"{titulo}\""}
                    ],
                    temperature=0.1,
                    max_tokens=30
                )

                categoria = response.choices[0].message.content.strip().title()

                if categoria.endswith('.'):
                    categoria = categoria[:-1]
                
                if not categoria or len(categoria) > 50:
                     print(f"  Aviso: LLM retornou resposta inválida/longa '{categoria}'. Usando 'Outros'.")
                     categoria = 'Outros'

                classificacoes.append(categoria)
                print(f"  -> Classificado como: {categoria}")
                break

            except Exception as e:
                tentativas += 1
                print(f"  Erro ao classificar título {i+1} (tentativa {tentativas}/{max_tentativas}): {e}")
                if tentativas >= max_tentativas:
                    print(f"  Falha final ao classificar título {i+1}. Usando 'Outros'.")
                    classificacoes.append('Outros')
                else:
                    print(f"  Aguardando {wait_time}s antes de tentar novamente...")
                    time.sleep(wait_time)
        
        time.sleep(1)

    return classificacoes

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Pacotes 'punkt' e/ou 'stopwords' do NLTK não encontrados. Baixando agora...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Falha ao tentar baixar pacotes NLTK: {e}. Continuando...")


DB_CONFIG = {
    'host': os.getenv("SCRAP_PG_HOST"),
    'port': os.getenv("SCRAP_PG_PORT"),
    'user': os.getenv("SCRAP_PG_USER"),
    'password': os.getenv("SCRAP_PG_PASSWORD"),
    'dbname': os.getenv("SCRAP_PG_DATABASE")
}
OUTPUT_DIR = "analises_graficos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")

def carregar_dados(conn):
    """Carrega todas as tabelas relevantes do banco de dados para DataFrames do Pandas."""
    print("Carregando dados do banco de dados...")
    query_noticias = "SELECT id, titulo, texto FROM scraping;"
    query_perguntas = "SELECT id, COALESCE(tipo_avaliacao, 'simples') as tipo_avaliacao, pergunta FROM perguntas;"
    query_respostas = "SELECT id, resposta FROM respostas;"

    df_noticias = pd.read_sql_query(query_noticias, conn)
    df_perguntas = pd.read_sql_query(query_perguntas, conn)
    df_respostas = pd.read_sql_query(query_respostas, conn)

    print(f"Dados carregados: {len(df_noticias)} notícias, {len(df_perguntas)} perguntas, {len(df_respostas)} respostas.")
    return df_noticias, df_perguntas, df_respostas

def plotar_estatisticas_gerais(df_noticias, df_perguntas, df_respostas):
    """Gera um gráfico de barras com as contagens totais do dataset."""
    print("Gerando gráfico de estatísticas gerais...")

    counts = {
        'Notícias': len(df_noticias),
        'Respostas': len(df_respostas)
    }

    tipo_counts = df_perguntas['tipo_avaliacao'].value_counts()

    tipo_legivel = {
        'multi_contexto': 'Perguntas (Multi-Contexto)',
        'rejeicao_negativa': 'Perguntas (Rejeição)',
        'simples': 'Perguntas (Simples)'
    }

    for tipo, count in tipo_counts.items():
        label = tipo_legivel.get(tipo, tipo)
        counts[label] = count

    ordered_labels = ['Notícias', 'Perguntas (Simples)', 'Perguntas (Multi-Contexto)', 'Perguntas (Rejeição)', 'Respostas']
    df_counts = pd.DataFrame([(label, counts.get(label, 0)) for label in ordered_labels], columns=['Entidade', 'Contagem'])
    df_counts = df_counts[df_counts['Contagem'] > 0]


    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df_counts.sort_values('Contagem', ascending=False), x='Contagem', y='Entidade')
    ax.set_xlabel('Quantidade', fontsize=12)
    ax.set_ylabel('Tipo de Dado', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_estatisticas_gerais.png'))
    plt.close()
    print("Gráfico 'estatisticas_gerais.png' salvo.")


def plotar_distribuicao_tamanho(df, coluna_texto, titulo, nome_arquivo):
    """Gera um histograma da distribuição do tamanho do texto."""
    print(f"Gerando histograma para '{titulo}'...")
    df_clean = df.dropna(subset=[coluna_texto])
    if df_clean.empty:
        print(f"Aviso: Não há dados para plotar em '{titulo}'. Pulando.")
        return

    df_clean['tamanho'] = df_clean[coluna_texto].str.len()

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=df_clean, x='tamanho', kde=True, bins=50)
    ax.set_xlabel('Número de Caracteres', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)

    mean_val = df_clean['tamanho'].mean()
    median_val = df_clean['tamanho'].median()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.0f}')
    plt.axvline(median_val, color='green', linestyle=':', label=f'Mediana: {median_val:.0f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, nome_arquivo))
    plt.close()
    print(f"Gráfico '{nome_arquivo}' salvo.")

def plotar_distribuicao_times(df_noticias):
    """Gera um gráfico de barras da distribuição de notícias por time."""
    print("Gerando gráfico de distribuição de times...")
    times_principais = ["Flamengo", "Corinthians", "Palmeiras", "São Paulo", "Vasco", "Fluminense", "Botafogo", "Grêmio", "Internacional", "Atlético-MG", "Cruzeiro"]

    contagem_times = Counter()
    for titulo in df_noticias['titulo']:
        if not isinstance(titulo, str): continue
        time_achado = None
        for time in times_principais:
             if time.lower() in titulo.lower():
                 time_achado = time
                 break
        if time_achado:
            contagem_times[time_achado] += 1

    if not contagem_times:
        print("Aviso: Nenhum time principal encontrado nos títulos. Pulando gráfico de times.")
        return

    df_times = pd.DataFrame(contagem_times.items(), columns=['Time', 'Quantidade de Notícias']).sort_values('Quantidade de Notícias', ascending=False)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_times, x='Quantidade de Notícias', y='Time')
    ax.set_xlabel('Quantidade de Notícias', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_distribuicao_times.png'))
    plt.close()
    print("Gráfico 'distribuicao_times.png' salvo.")


def gerar_nuvem_palavras(df_noticias):
    """Gera uma nuvem de palavras com os termos mais frequentes dos títulos."""
    print("Gerando nuvem de palavras dos títulos...")
    stopwords_pt = set(nltk.corpus.stopwords.words('portuguese'))
    stopwords_pt.update(['ge', 'veja', 'contra', 'após', 'sobre', 'diz', 'pode', 'ser', 'faz', 'tudo', 'vai', 'aos', 'nas', 'nos', 'pelo', 'pela'])

    textos_validos = df_noticias['titulo'].dropna().astype(str)
    texto_completo = ' '.join(textos_validos)

    if not texto_completo.strip():
        print("Aviso: Não há texto nos títulos para gerar a nuvem de palavras. Pulando.")
        return

    texto_completo = texto_completo.lower()

    try:
        wordcloud = WordCloud(width=1200, height=600,
                            background_color='white',
                            stopwords=stopwords_pt,
                            min_font_size=10,
                            max_words=150,
                            collocations=False
                            ).generate(texto_completo)

        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, '06_nuvem_palavras_titulos.png'))
        plt.close()
        print("Gráfico 'nuvem_palavras_titulos.png' salvo.")
    except ValueError as e:
         print(f"Erro ao gerar nuvem de palavras (provavelmente texto insuficiente após stopwords): {e}. Pulando.")


def plotar_distribuicao_esportes(df_noticias, llm_client: OpenAI):
    """
    Classifica notícias por esporte usando LLM e gera um gráfico de barras.
    """
    print("Iniciando classificação de esportes via LLM...")

    titulos = df_noticias['titulo'].tolist()

    categorias_llm = categorizar_esporte_llm(titulos, llm_client)

    df_noticias['esporte_llm'] = categorias_llm

    esporte_counts = df_noticias['esporte_llm'].value_counts()

    if esporte_counts.empty:
        print("Nenhuma categoria de esporte foi classificada pelo LLM. Pulando gráfico.")
        return

    df_esportes = esporte_counts.reset_index()
    df_esportes.columns = ['Esporte', 'Quantidade']

    top_n = 20
    if len(df_esportes) > top_n:
        df_top = df_esportes.nlargest(top_n, 'Quantidade')
        outros_sum = df_esportes.nsmallest(len(df_esportes) - top_n, 'Quantidade')['Quantidade'].sum()
        df_outros = pd.DataFrame([{'Esporte': 'Outros (LLM)', 'Quantidade': outros_sum}])
        df_plot = pd.concat([df_top, df_outros], ignore_index=True)
    else:
        df_plot = df_esportes

    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=df_plot.sort_values('Quantidade', ascending=False),
                     x='Quantidade', y='Esporte')

    ax.set_xlabel('Quantidade de Notícias', fontsize=12)
    ax.set_ylabel('Esporte', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fontsize=10, padding=3)

    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_distribuicao_esportes_llm.png'))
    plt.close()
    print("Gráfico '07_distribuicao_esportes_llm.png' salvo.")


def main():
    """Função principal para executar a análise."""
    conn = None
    llm_client = None

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("A variável de ambiente OPENAI_API_KEY não foi definida.")
        llm_client = OpenAI(api_key=openai_api_key)
        print("Cliente OpenAI inicializado.")

        conn = psycopg2.connect(**DB_CONFIG)
        if conn:
             print("Conexão com o banco de dados estabelecida.")
        else:
             print("Falha ao conectar ao banco de dados.")
             return

        df_noticias, df_perguntas, df_respostas = carregar_dados(conn)

        # Gerar gráficos (descomente os que quiser)
        # plotar_estatisticas_gerais(df_noticias, df_perguntas, df_respostas)
        # plotar_distribuicao_tamanho(df_noticias, 'texto', 'Distribuição do Tamanho das Notícias', '02_dist_tamanho_noticias.png')
        # plotar_distribuicao_tamanho(df_perguntas, 'pergunta', 'Distribuição do Tamanho das Perguntas', '03_dist_tamanho_perguntas.png')
        # plotar_distribuicao_tamanho(df_respostas, 'resposta', 'Distribuição do Tamanho das Respostas', '04_dist_tamanho_respostas.png')
        # plotar_distribuicao_times(df_noticias)
        # gerar_nuvem_palavras(df_noticias)
        # plotar_distribuicao_esportes(df_noticias.copy(), llm_client)

        print(f"\nAnálise concluída! Todos os gráficos foram salvos na pasta '{OUTPUT_DIR}'.")

    except psycopg2.OperationalError as e:
        print(f"Erro de conexão com o banco de dados: {e}")
        print("Verifique as variáveis de ambiente DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME.")
    except psycopg2.Error as e:
        print(f"Ocorreu um erro de banco de dados: {e}")
    except ValueError as e:
         print(f"Erro de configuração: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado no script: {e}")
    finally:
        if conn:
            conn.close()
            print("\nConexão com o banco de dados fechada.")

if __name__ == "__main__":
    main()