import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

def get_db_connection(db_env_vars):
    """Cria uma conexão psycopg2 a partir de um dicionário de variáveis de ambiente."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv(db_env_vars['db']),
            user=os.getenv(db_env_vars['user']),
            password=os.getenv(db_env_vars['pass']),
            host=os.getenv(db_env_vars['host']),
            port=os.getenv(db_env_vars['port'])
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"ERRO: Não foi possível conectar ao banco de dados {db_env_vars['db']}: {e}", file=sys.stderr)
        return None

def get_question_types(scrap_db_params):
    """Busca o 'tipo_avaliacao' de todas as perguntas do banco de scraping."""
    print("-> Carregando tipos de perguntas do banco de dados de scraping...")
    conn = get_db_connection(scrap_db_params)
    if not conn:
        sys.exit("ERRO FATAL: Não foi possível conectar ao WSCARP_DB.")

    question_types = {}
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT id, tipo_avaliacao FROM perguntas")
            records = cursor.fetchall()
            for record in records:
                if record['tipo_avaliacao'] == 'multi_contexto':
                    question_types[record['id']] = 'Integracao'
                elif record['tipo_avaliacao'] == 'rejeicao_negativa':
                    question_types[record['id']] = 'Rejeicao'
                else:
                    # Considera NULO como "Factual Simples"
                    question_types[record['id']] = 'Factual_Simples'
        
        print(f"   - {len(question_types)} tipos de perguntas carregados.")
        return question_types
    except Exception as e:
        print(f"   - ERRO ao buscar tipos de perguntas: {e}", file=sys.stderr)
        return None
    finally:
        if conn:
            conn.close()

def fetch_all_results(architectures, question_types):
    """Busca todos os resultados de todos os bancos e os consolida em um DataFrame."""
    all_data = []
    
    for arch in architectures:
        print(f"-> Buscando dados da arquitetura: {arch['name']}...")
        conn = get_db_connection(arch['params'])
        if not conn:
            continue
            
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if arch['is_baseline']:
                    cursor.execute("SELECT question_id, llm_judge_score FROM evaluation_results")
                else:
                    cursor.execute("""
                        SELECT question_id, llm_judge_score, ragas_faithfulness, 
                               ragas_context_relevancy AS context_precision, 
                               ragas_context_recall
                        FROM evaluation_results
                    """)
                
                records = cursor.fetchall()
                
                for record in records:
                    q_id = record['question_id']
                    tipo_avaliacao = question_types.get(q_id)
                    if not tipo_avaliacao:
                        continue

                    row = {
                        'Arquitetura': arch['name'],
                        'Tipo_Avaliacao': tipo_avaliacao,
                        'llm_judge_score': record.get('llm_judge_score'),
                        'faithfulness': record.get('ragas_faithfulness'),
                        'context_precision': record.get('context_precision'),
                        'context_recall': record.get('ragas_context_recall')
                    }
                    all_data.append(row)
            
            print(f"   - {len(records)} registros encontrados para {arch['name']}.")

        except Exception as e:
            print(f"   - ERRO ao buscar dados de {arch['name']}: {e}", file=sys.stderr)
        finally:
            if conn:
                conn.close()
                
    return pd.DataFrame(all_data)

def print_styled_table(df, title):
    """Imprime uma tabela formatada de Média (Desvio Padrão)."""
    
    def format_mean_std(x):
        mean = x['mean']
        std = x['std']
        if pd.isna(mean) or pd.isna(std):
            return "N/A"
        return f"{mean:.4f} (±{std:.4f})"

    grouped = df.groupby('Arquitetura').agg(['mean', 'std'])
    
    formatted_df = pd.DataFrame(index=grouped.index)
    for col in df.columns.drop('Arquitetura'):
        if (col, 'mean') in grouped.columns:
            formatted_df[col] = grouped[col].apply(format_mean_std, axis=1)
            
    col_order = [
        'llm_judge_score', 
        'faithfulness', 
        'context_precision', 
        'context_recall'
    ]
    existing_cols = [c for c in col_order if c in formatted_df.columns]
    formatted_df = formatted_df[existing_cols]
    
    formatted_df = formatted_df.rename(columns={
        'llm_judge_score': 'LLM Judge',
        'faithfulness': 'Faithfulness',
        'context_precision': 'Context Precision',
        'context_recall': 'Context Recall'
    })

    print("\n\n" + "="*80)
    print(f"RESULTADOS: {title}")
    print("="*80)
    print(formatted_df.to_markdown(numalign="left", stralign="left"))
    print("\nValores: Média (± Desvio Padrão)")

def setup_plot_style():
    """Configura o estilo global dos gráficos."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_quality_factual(df_factual):
    """Gráfico 1: Qualidade Factual Média com Desvio Padrão (Barplot) - Sem Título."""
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=df_factual,
        x='Arquitetura',
        y='llm_judge_score',
        errorbar='sd',
        capsize=.1,
        palette='viridis',
        hue='Arquitetura',
        legend=False
    )
    plt.ylabel('Accuracy Média', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 5.5)
    
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.2f', 
            padding=3, 
            fontsize=11, 
            fontweight='bold',
            bbox={"boxstyle": "round,pad=0.2", 
                  "facecolor": "white", 
                  "edgecolor": "none", 
                  "alpha": 0.8}
        )

    plt.tight_layout()
    plt.savefig('plots/1_qualidade_factual_media_v2.png', dpi=300) 
    plt.close()
    print("   - Gráfico 1 salvo (sem título e com rótulos protegidos contra grid).")

def plot_pipeline_metrics(df_factual):
    """Gráfico 2: Métricas de Pipeline do RAG (Grouped Barplot) - Sem Título."""
    df_rag = df_factual[df_factual['Arquitetura'] != 'Baseline LLM'].copy()
    df_melted = df_rag.melt(
        id_vars=['Arquitetura'],
        value_vars=['faithfulness', 'context_precision', 'context_recall'],
        var_name='Métrica',
        value_name='Score'
    )
    metric_labels = {
        'faithfulness': 'Faithfulness',
        'context_precision': 'Precision',
        'context_recall': 'Recall'
    }
    df_melted['Métrica'] = df_melted['Métrica'].map(metric_labels)

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df_melted,
        x='Métrica',
        y='Score',
        hue='Arquitetura',
        errorbar='sd',
        capsize=.05,
        palette='viridis'
    )
    plt.ylabel('Pontuação', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Arquitetura', title_fontsize=12, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('plots/2_metricas_pipeline.png', dpi=300)
    plt.close()
    print("   - Gráfico 2 salvo (sem título).")

def plot_quality_distribution(df_factual):
    """
    Gráfico 3: Distribuição de Qualidade (Grouped Bar Chart de Contagens) - Sem Título.
    Mostra a quantidade de ocorrências de cada llm_judge_score (1 a 5) por Arquitetura.
    """
    print("-> Gerando Gráfico 3: Distribuição de Qualidade (Contagem de Scores)...")
    
    df_factual_clean = df_factual.copy()
    df_factual_clean['llm_judge_score'] = df_factual_clean['llm_judge_score'].round().clip(1, 5).astype('Int64')
    
    score_counts = pd.crosstab(
        df_factual_clean['Arquitetura'], 
        df_factual_clean['llm_judge_score'], 
        rownames=['Arquitetura'], 
        colnames=['Score']
    )
    
    score_counts.columns = score_counts.columns.astype(str)
    
    df_plot = score_counts.stack().reset_index(name='Contagem')
    
    df_plot['Score'] = df_plot['Score'].astype(int) 

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df_plot,
        x='Arquitetura',
        y='Contagem',
        hue='Score',
        palette='viridis',
        hue_order=sorted(df_plot['Score'].unique())
    )
    
    plt.ylabel('Contagem de Ocorrências', fontsize=14)
    plt.xlabel('Arquitetura', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.0f',
            padding=3, 
            fontsize=9, 
            fontweight='bold', 
            bbox={"boxstyle": "round,pad=0.2", 
                  "facecolor": "white", 
                  "edgecolor": "none", 
                  "alpha": 0.8}
        )

    plt.legend(
        title='LLM Judge Score', 
        title_fontsize=12, 
        fontsize=11, 
        loc='upper center', 
        ncol=5,
        frameon=True, 
        bbox_to_anchor=(0.5, 1.15) 
    )

    plt.tight_layout()
    plt.savefig('plots/3_distribuicao_qualidade_contagem.png', dpi=300)
    plt.close()
    print("   - Gráfico 3 (Grouped Bar Chart de Contagens) salvo: plots/3_distribuicao_qualidade_contagem.png")

def plot_radar_chart(df_factual):
    """Gráfico 4: Perfil das Arquiteturas (Radar Chart) - Sem Título e com Ênfase nas Categorias."""
    df_rag_means = df_factual[df_factual['Arquitetura'] != 'Baseline LLM'].groupby('Arquitetura')[
        ['faithfulness', 'context_precision', 'context_recall']
    ].mean()

    categories = ['Faithfulness', 'Precision', 'Recall']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='black', size=14, weight='bold')

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)

    colors = sns.color_palette('viridis', n_colors=3)
    archs = df_rag_means.index.tolist()

    for i, arch in enumerate(archs):
        values = df_rag_means.loc[arch].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=arch, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.tick_params(axis='x', pad=20)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/4_radar_pipeline_v2.png', dpi=300)
    plt.close()
    print("   - Gráfico 4 salvo (versão com ênfase nas categorias).")

def plot_factuality_vs_safety(df_all_results):
    """Gráfico 5: Trade-off Factualidade vs. Segurança (Scatter Plot Melhorado)."""
    factual_means = df_all_results[
        df_all_results['Tipo_Avaliacao'] == 'Factual_Simples'
    ].groupby('Arquitetura')['llm_judge_score'].mean()

    safety_means = df_all_results[
        df_all_results['Tipo_Avaliacao'] == 'Rejeicao'
    ].groupby('Arquitetura')['llm_judge_score'].mean()

    df_tradeoff = pd.DataFrame({
        'Desempenho Factual': factual_means,
        'Segurança': safety_means
    }).reset_index()

    sns.set_theme(style="white", font_scale=1.1)
    plt.figure(figsize=(10, 8))
    
    plt.axvspan(3.0, 5.5, ymin=3.0/5.5, ymax=1, color='green', alpha=0.05, zorder=0)
    plt.text(5.4, 5.4, 'Objetivo Ideal', ha='right', va='top', 
             fontsize=12, color='green', alpha=0.4, weight='bold')

    plt.grid(True, linestyle=':', color='gray', alpha=0.3, zorder=0)

    ax = sns.scatterplot(
        data=df_tradeoff,
        x='Desempenho Factual',
        y='Segurança',
        hue='Arquitetura',
        style='Arquitetura',
        s=600,
        palette='viridis',
        markers=['o', 's', 'D', '^'],
        edgecolor='white',
        linewidth=2,
        zorder=3
    )

    for i in range(df_tradeoff.shape[0]):
        arch = df_tradeoff['Arquitetura'][i]
        x_pos = df_tradeoff['Desempenho Factual'][i]
        y_pos = df_tradeoff['Segurança'][i]
        
        xf = 0.08
        yf = 0.08
        
        if arch == 'Advanced RAG': yf += 0.05 
        if arch == 'Naive RAG': xf += 0.05
        
        plt.text(
            x_pos + xf,
            y_pos + yf,
            arch,
            fontdict={'weight': 'medium', 'size': 11},
            va='center',
            zorder=4
        )

    plt.xlabel('Desempenho Factual Médio', fontsize=13)
    plt.ylabel('Segurança Média', fontsize=13)
    
    plt.xlim(1.0, 5.5)
    plt.ylim(1.0, 5.5)
    
    sns.despine(trim=True, offset=10)
    
    plt.tight_layout()
    plt.savefig('plots/5_factualidade_vs_seguranca_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Gráfico 5 (versão limpa) salvo: plots/5_factualidade_vs_seguranca_v2.png")

def generate_all_plots(df_all_results):
    """Função orquestradora para gerar todos os gráficos."""
    print("\n--- Iniciando Geração de Gráficos ---")
    setup_plot_style()

    df_factual = df_all_results[df_all_results['Tipo_Avaliacao'] != 'Rejeicao'].copy()
    
    plot_quality_factual(df_factual)
    plot_pipeline_metrics(df_factual)
    plot_quality_distribution(df_factual)
    plot_radar_chart(df_factual)
    plot_factuality_vs_safety(df_all_results)
    
    print("--- Geração de Gráficos Concluída (ver pasta 'plots/') ---")

def main():
    load_dotenv()

    scrap_db_params = {
        'db': 'SCRAP_PG_DATABASE', 'user': 'SCRAP_PG_USER', 'pass': 'SCRAP_PG_PASSWORD',
        'host': 'SCRAP_PG_HOST', 'port': 'SCRAP_PG_PORT'
    }
    architectures = [
        {"params": {'db': 'BASELINE_PG_DATABASE', 'user': 'BASELINE_PG_USER', 'pass': 'BASELINE_PG_PASSWORD', 'host': 'BASELINE_PG_HOST', 'port': 'BASELINE_PG_PORT'}, "name": "Baseline LLM", "is_baseline": True},
        {"params": {'db': 'PG_DATABASE', 'user': 'PG_USER', 'pass': 'PG_PASSWORD', 'host': 'PG_HOST', 'port': 'PG_PORT'}, "name": "Naive RAG", "is_baseline": False},
        {"params": {'db': 'ADV_PG_DATABASE', 'user': 'ADV_PG_USER', 'pass': 'ADV_PG_PASSWORD', 'host': 'ADV_PG_HOST', 'port': 'ADV_PG_PORT'}, "name": "Advanced RAG", "is_baseline": False},
        {"params": {'db': 'GRAPH_PG_DATABASE', 'user': 'GRAPH_PG_USER', 'pass': 'GRAPH_PG_PASSWORD', 'host': 'GRAPH_PG_HOST', 'port': 'GRAPH_PG_PORT'}, "name": "Graph RAG", "is_baseline": False},
    ]

    question_types = get_question_types(scrap_db_params)
    if not question_types:
        sys.exit(1)

    df_all_results = fetch_all_results(architectures, question_types)
    if df_all_results.empty:
        print("Nenhum dado de avaliação foi encontrado. Abortando.", file=sys.stderr)
        sys.exit(1)
    
    df_factual = df_all_results[df_all_results['Tipo_Avaliacao'] == 'Factual_Simples']
    print_styled_table(df_factual.drop(columns=['Tipo_Avaliacao']), "Perguntas Fatuais (Simples e Multi-Contexto)")
    
    df_rejection = df_all_results[df_all_results['Tipo_Avaliacao'] == 'Rejeicao']
    print_styled_table(df_rejection.drop(columns=['Tipo_Avaliacao', 'faithfulness', 'context_precision', 'context_recall']), "Testes de Rejeição Negativa")

    generate_all_plots(df_all_results)
    
    print("\n\n--- Análise Concluída ---")


if __name__ == "__main__":
    main()