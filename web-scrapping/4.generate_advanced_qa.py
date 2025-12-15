import os
import psycopg2
import json
from openai import OpenAI
from dotenv import load_dotenv
from itertools import combinations
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

DB_CONFIG = {
    'host': os.getenv("SCRAP_PG_HOST"),
    'port': os.getenv("SCRAP_PG_PORT"),
    'user': os.getenv("SCRAP_PG_USER"),
    'password': os.getenv("SCRAP_PG_PASSWORD"),
    'dbname': os.getenv("SCRAP_PG_DATABASE")
}

client = OpenAI(api_key=openai_api_key)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

GENERATION_MODEL = "gpt-3.5-turbo"

MULTI_CONTEXT_PROMPT_QUESTION = """
Você é um especialista em análise de texto e criação de perguntas complexas para avaliação de sistemas de IA (RAG).
Sua tarefa é criar UMA ÚNICA pergunta que SÓ possa ser respondida combinando informações de TODOS os textos fornecidos abaixo.
A pergunta não deve ser respondível usando apenas um dos textos isoladamente. Ela deve forçar a síntese de informações.
**Critérios para a pergunta:**
1.  Deve ser clara, objetiva e sobre um fato ou evento específico.
2.  Deve conectar explicitamente conceitos ou entidades de ambos os textos.
3.  Evite perguntas de "sim/não".
**Formato da Resposta:**
Sua resposta deve ser APENAS um objeto JSON válido com a chave "pergunta". Exemplo: {{"pergunta": "Qual foi o impacto da atuação do jogador X no jogo Y, considerando sua recente recuperação de lesão mencionada no outro artigo?"}}
**Textos Fornecidos:**
---
**Texto 1:**
{texto_1}
---
**Texto 2:**
{texto_2}
---
"""

MULTI_CONTEXT_PROMPT_ANSWER = """
Você é um assistente de IA especialista em responder perguntas com base em múltiplos documentos.
Sua tarefa é responder à pergunta fornecida, sintetizando as informações de TODOS os textos de contexto de forma coesa e completa.
**Contextos Fornecidos:**
---
**Texto 1:**
{texto_1}
---
**Texto 2:**
{texto_2}
---
**Pergunta:**
{pergunta}
---
**Resposta Sintetizada:**
"""

NEGATIVE_REJECTION_PROMPT_QUESTION = """
Você é um especialista em criar perguntas para testar a robustez de sistemas de IA (RAG).
Sua tarefa é gerar UMA ÚNICA pergunta sobre notícias esportivas que seja plausível, mas cuja resposta NÃO esteja contida no resumo de tópicos fornecido.
A pergunta deve parecer que poderia estar nas notícias, mas ser sobre um evento, jogador ou detalhe que você tem certeza que não está listado.
**Critérios para a pergunta:**
1.  Deve ser sobre o universo de esportes, preferencialmente futebol brasileiro.
2.  Pode ser sobre um evento fictício, uma estatística inventada, ou um jogador não mencionado.
3.  A pergunta deve ser específica para dificultar respostas genéricas.
**Formato da Resposta:**
Sua resposta deve ser APENAS um objeto JSON válido com a chave "pergunta".
**Resumo dos Tópicos Existentes na Base de Conhecimento:**
{lista_topicos}
"""

def gerar_pergunta_llm(prompt, model=GENERATION_MODEL):
    """Gera a pergunta e retorna o texto, com tratamento de erro e log robustos."""
    response_content = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        data = json.loads(response_content)
        
        if "pergunta" not in data:
            print(f"  -> Erro de Chave: A chave 'pergunta' não foi encontrada no JSON retornado.")
            print(f"     Resposta recebida da API: {response_content}")
            return None
        
        pergunta = data.get("pergunta")
        if not pergunta or not pergunta.strip():
            print(f"  -> Erro de Conteúdo: A chave 'pergunta' foi encontrada, mas está vazia.")
            print(f"     Resposta recebida da API: {response_content}")
            return None

        return pergunta
        
    except json.JSONDecodeError:
        print(f"  -> Erro de Decodificação: A resposta da API não é um JSON válido.")
        print(f"     Resposta recebida que causou o erro: {response_content}")
        return None
    except KeyError as e:
        print(f"  -> Erro de Chave Inesperado: {e}")
        print(f"     Resposta recebida da API: {response_content}")
        return None
    except Exception as e:
        print(f"  -> Erro inesperado na função gerar_pergunta_llm: {e}")
        if response_content:
            print(f"     Resposta recebida da API (se houver): {response_content}")
        return None

def gerar_resposta_llm(prompt, model=GENERATION_MODEL):
    """Gera a resposta e retorna o texto."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  -> Erro ao gerar resposta com LLM: {e}")
        return None

def gerar_embedding(texto):
    """Gera o embedding para um texto."""
    try:
        return embeddings_model.embed_query(texto)
    except Exception as e:
        print(f"  -> Erro ao gerar embedding: {e}")
        return None

def generate_multi_context_qa(cur, conn, num_perguntas=30):
    """Gera pares de Pergunta-Resposta que requerem múltiplos contextos."""
    print("\n--- Iniciando geração de P&R MULTI-CONTEXTO ---")
    
    cur.execute("SELECT id, titulo, texto FROM scraping WHERE texto IS NOT NULL AND LENGTH(texto) > 200;")
    noticias = cur.fetchall()
    
    noticias_por_time = defaultdict(list)
    times_principais = ["Flamengo", "Corinthians", "Palmeiras", "São Paulo", "Vasco", "Fluminense", "Botafogo", "Grêmio", "Internacional", "Atlético-MG", "Cruzeiro"]
    for id, titulo, texto in noticias:
        for time in times_principais:
            if time.lower() in titulo.lower():
                noticias_por_time[time].append({'id': id, 'texto': texto})
                break
    
    perguntas_geradas = 0
    
    for time, items in noticias_por_time.items():
        if perguntas_geradas >= num_perguntas: break
        if len(items) >= 2:
            for combo in combinations(items, 2):
                if perguntas_geradas >= num_perguntas: break
                
                noticia1, noticia2 = combo[0], combo[1]
                print(f"Processando combinação de notícias {noticia1['id']} e {noticia2['id']}...")

                pergunta = None
                try:
                    print("  [DEBUG] Etapa 1: Criando o prompt da pergunta...")
                    prompt_pergunta = MULTI_CONTEXT_PROMPT_QUESTION.format(texto_1=noticia1['texto'], texto_2=noticia2['texto'])
                    
                    print("  [DEBUG] Etapa 2: Chamando a função gerar_pergunta_llm...")
                    pergunta = gerar_pergunta_llm(prompt_pergunta)
                    print("  [DEBUG] Etapa 3: Retorno da função recebido.")

                except Exception as e:
                    print(f"\n  !!!!!! ERRO FATAL CAPTURADO AQUI !!!!!!")
                    print(f"  O erro foi do tipo: {type(e).__name__}")
                    print(f"  Mensagem de erro: {e}")
                    print(f"  Isso indica que o problema ocorreu ANTES ou DURANTE a chamada da API, fora do try/except interno da função.\n")
                    raise e
                
                if not pergunta:
                    print("  -> Falha na geração da pergunta ou pergunta vazia. Pulando para a próxima combinação.")
                    continue

                print("  [DEBUG] Etapa 4: Gerando embedding da pergunta...")
                pergunta_embedding = gerar_embedding(pergunta)
                if not pergunta_embedding: continue
                
                print("  [DEBUG] Etapa 5: Inserindo pergunta no banco de dados...")
                cur.execute(
                    """
                    INSERT INTO perguntas (scraping_id, pergunta, embedding, tipo_avaliacao, contexto_ids) 
                    VALUES (%s, %s, %s, %s, %s) RETURNING id
                    """,
                    (noticia1['id'], pergunta, pergunta_embedding, 'multi_contexto', [noticia1['id'], noticia2['id']])
                )
                pergunta_id = cur.fetchone()[0]
                
                print("  [DEBUG] Etapa 6: Gerando a resposta...")
                prompt_resposta = MULTI_CONTEXT_PROMPT_ANSWER.format(texto_1=noticia1['texto'], texto_2=noticia2['texto'], pergunta=pergunta)
                resposta = gerar_resposta_llm(prompt_resposta)
                if not resposta: continue

                print("  [DEBUG] Etapa 7: Gerando embedding da resposta...")
                resposta_embedding = gerar_embedding(resposta)
                if not resposta_embedding: continue

                print("  [DEBUG] Etapa 8: Inserindo resposta no banco de dados...")
                cur.execute(
                    "INSERT INTO respostas (pergunta_id, resposta, embedding) VALUES (%s, %s, %s)",
                    (pergunta_id, resposta, resposta_embedding)
                )
                conn.commit()
                perguntas_geradas += 1
                print(f"  -> Sucesso! P&R {perguntas_geradas}/{num_perguntas} gerado e salvo.")

    print(f"--- Geração de {perguntas_geradas} P&R MULTI-CONTEXTO concluída! ---")

def generate_negative_rejection_qa(cur, conn, num_perguntas=15):
    """Gera perguntas sem resposta na base e insere a resposta padrão."""
    print("\n--- Iniciando geração de P&R de REJEIÇÃO NEGATIVA ---")
    
    cur.execute("SELECT titulo FROM scraping;")
    titulos = cur.fetchall()
    lista_topicos = "- " + "\n- ".join([t[0] for t in titulos])
    
    resposta_padrao = "A informação necessária para responder a esta pergunta não foi encontrada nos documentos fornecidos."
    resposta_embedding = gerar_embedding(resposta_padrao)

    perguntas_geradas = 0
    while perguntas_geradas < num_perguntas:
        print(f"Tentando gerar pergunta de rejeição {perguntas_geradas + 1}/{num_perguntas}...")
        prompt = NEGATIVE_REJECTION_PROMPT_QUESTION.format(lista_topicos=lista_topicos)
        pergunta = gerar_pergunta_llm(prompt)
        
        if pergunta:
            pergunta_embedding = gerar_embedding(pergunta)
            if not pergunta_embedding: continue

            cur.execute(
                """
                INSERT INTO perguntas (scraping_id, pergunta, embedding, tipo_avaliacao) 
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (None, pergunta, pergunta_embedding, 'rejeicao_negativa')
            )
            pergunta_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO respostas (pergunta_id, resposta, embedding) VALUES (%s, %s, %s)",
                (pergunta_id, resposta_padrao, resposta_embedding)
            )
            conn.commit()
            perguntas_geradas += 1
            print(f"  -> Sucesso! P&R {perguntas_geradas}/{num_perguntas} gerado e salvo.")

    print(f"--- Geração de {perguntas_geradas} P&R de REJEIÇÃO NEGATIVA concluída! ---")

def main():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        generate_multi_context_qa(cur, conn, num_perguntas=30)
        generate_negative_rejection_qa(cur, conn, num_perguntas=15)

    except psycopg2.Error as e:
        print(f"Ocorreu um erro de banco de dados: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado no script: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("\nConexão com o banco de dados fechada.")

if __name__ == "__main__":
    main()