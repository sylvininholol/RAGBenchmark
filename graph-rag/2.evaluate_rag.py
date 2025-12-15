import os
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from openai import OpenAI, RateLimitError

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from retriever import retrieve_context_from_graph
from generate_response import generate_response
from llm_judge import evaluate_with_llm_judge

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextRecall,
    ContextPrecision,
)
from datasets import Dataset

def fetch_questions_for_evaluation(scraping_cursor):
    """Busca perguntas e respostas 'padrão ouro' do banco de scrapping."""
    print("-> Buscando TODAS as perguntas 'padrão ouro' para avaliação...")
    query = """
        SELECT p.id, p.pergunta, r.resposta AS ground_truth_answer, r.embedding AS ground_truth_embedding
        FROM perguntas p
        JOIN respostas r ON p.id = r.pergunta_id
        WHERE p.embedding IS NOT NULL AND r.embedding IS NOT NULL AND r.resposta IS NOT NULL;
    """
    
    scraping_cursor.execute(query)
    questions = scraping_cursor.fetchall()
    print(f"   - {len(questions)} pares de pergunta/resposta encontrados para avaliação.")
    return questions

def calculate_cosine_similarity(embedding1, embedding2) -> float:
    """Calcula a similaridade de cosseno entre dois embeddings."""
    import torch
    import torch.nn.functional as F
    try:
        emb1 = json.loads(embedding1) if isinstance(embedding1, str) else embedding1
        emb2 = json.loads(embedding2) if isinstance(embedding2, str) else embedding2
        tensor1 = torch.tensor(emb1).unsqueeze(0)
        tensor2 = torch.tensor(emb2).unsqueeze(0)
        return F.cosine_similarity(tensor1, tensor2).item()
    except Exception:
        return 0.0


def run_rag_pipeline_for_question(item: dict, neo4j_driver, embeddings_model, translator_llm, final_llm) -> dict:
    """
    Executa o pipeline Graph RAG (recuperação + geração) para uma única pergunta
    e retorna todos os artefatos necessários para a avaliação.
    [CORRIGIDO COM RETRY PARA RATE LIMIT]
    """
    question_id = item['id']
    question_text = item['pergunta']
    ground_truth_answer = item['ground_truth_answer']
    ground_truth_embedding = item['ground_truth_embedding']

    print(f"\n--- Processando Pergunta ID: {question_id} ---")
    print(f"   Pergunta: {question_text}")

    max_retries = 3
    retries = 0
    wait_time = 60

    while retries < max_retries:
        try:
            retrieved_context = retrieve_context_from_graph(question_text, neo4j_driver, embeddings_model, translator_llm)
            
            if "Nenhum contexto" in retrieved_context or "Erro" in retrieved_context or not retrieved_context:
                 rag_answer_text = "Não foi possível responder com base nas evidências do conhecimento disponível."
            else:
                 rag_answer_text = generate_response(retrieved_context, question_text, final_llm)

            rag_answer_embedding = embeddings_model.embed_query(rag_answer_text)
            similarity_score = calculate_cosine_similarity(ground_truth_embedding, rag_answer_embedding)
            print(f"   - [ID {question_id}] Similaridade de Cosseno: {similarity_score:.4f}")

            return {
                "question_id": question_id,
                "question": question_text,
                "contexts": [retrieved_context],
                "answer": rag_answer_text,
                "ground_truth": ground_truth_answer,
                "rag_answer_embedding": str(rag_answer_embedding),
                "ground_truth_answer_embedding": str(ground_truth_embedding),
                "cosine_similarity": similarity_score,
                "model_name": final_llm.model_name
            }
        
        except RateLimitError as e:
            retries += 1
            print(f"   - ERRO (RateLimit) ao processar ID {question_id} (Tentativa {retries}/{max_retries}): {e}. Pausando por {wait_time}s.")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"   - ERRO (Geral) ao processar ID {question_id}: {e}")
            return {"question_id": question_id, "error": f"Exception: {e}"}
    
    print(f"   - ERRO FATAL (RateLimit) ao processar ID {question_id}: Excedidas {max_retries} tentativas.")
    return {"question_id": question_id, "error": f"Excedidas {max_retries} tentativas de RateLimit"}


def run_evaluation(rag_conn, scraping_cursor, neo4j_driver, embeddings_model, translator_llm, final_llm, openai_api_key: str):
    """Orquestra o processo de avaliação, salvando no DB."""
    
    questions_to_evaluate = fetch_questions_for_evaluation(scraping_cursor)
    eval_cursor = rag_conn.cursor()
    eval_cursor.execute("SELECT question_id FROM evaluation_results")
    evaluated_ids = {row[0] for row in eval_cursor.fetchall()}
    print(f"   - {len(evaluated_ids)} perguntas já avaliadas e serão puladas.")
    
    questions_to_process = [item for item in questions_to_evaluate if item['id'] not in evaluated_ids]

    if not questions_to_process:
        print("-> Todas as perguntas (deste bloco) já foram avaliadas.")
        eval_cursor.close()
        return

    print(f"-> Fase 1: Executando o pipeline de RAG para {len(questions_to_process)} perguntas em paralelo...")
    
    pipeline_results = []
    MAX_WORKERS = 5
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(run_rag_pipeline_for_question, item, neo4j_driver, embeddings_model, translator_llm, final_llm): item
            for item in questions_to_process
        }

        for future in as_completed(future_to_question):
            try:
                result_data = future.result()
                if "error" not in result_data:
                    pipeline_results.append(result_data)
                    print(f"   - [ID {result_data['question_id']}] Pipeline RAG concluído.")
                else:
                    item_failed = future_to_question[future]
                    print(f"   - [ID {item_failed['id']}] Pipeline RAG falhou: {result_data['error']}")
            except Exception as exc:
                question_item = future_to_question[future]
                print(f"   - ERRO ao processar pipeline da pergunta ID {question_item['id']}: {exc}")

    if not pipeline_results:
        print("\n--- Nenhuma pergunta foi processada com sucesso. Encerrando avaliação. ---")
        eval_cursor.close()
        return

    print("\n\n-> Fase 2: Preparando e executando a avaliação em lote com RAGAS...")
    for r in pipeline_results:
        r['reference'] = r['ground_truth']
    ragas_dataset = Dataset.from_list(pipeline_results)

    metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]

    ragas_scores_df = None
    try:
        ragas_result = evaluate(ragas_dataset, metrics=metrics, raise_exceptions=False)
        print("   - Avaliação com RAGAS concluída.")
        print(ragas_result)
        ragas_scores_df = ragas_result.to_pandas()
    except Exception as e:
        print(f"ERRO: Avaliação RAGAS falhou: {e}. Scores RAGAS serão nulos.")


    print("\n-> Fase 3: Executando LLM-as-Judge e salvando no PostgreSQL...")
    
    for i, pipeline_res in enumerate(pipeline_results):
        question_id = pipeline_res["question_id"]
        print(f"   - [ID {question_id}] Avaliando e salvando...")
        
        ragas_scores = {}
        if ragas_scores_df is not None:
            try:
                ragas_scores = ragas_scores_df.iloc[i].to_dict()
            except Exception as e:
                print(f"   - AVISO: Falha ao mapear scores RAGAS para ID {question_id}: {e}")
        
        judge_score, judge_reasoning = evaluate_with_llm_judge(
            question=pipeline_res["question"],
            ground_truth_answer=pipeline_res["ground_truth"],
            generated_answer=pipeline_res["answer"],
            api_key=openai_api_key
        )
        
        final_result = {
            "question_id": question_id,
            "question_text": pipeline_res["question"],
            "ground_truth_answer_embedding": pipeline_res["ground_truth_answer_embedding"],
            "rag_answer_text": pipeline_res["answer"],
            "rag_answer_embedding": pipeline_res["rag_answer_embedding"],
            "retrieved_context": "\n---\n".join(pipeline_res["contexts"]),
            "cosine_similarity": pipeline_res["cosine_similarity"],
            "llm_judge_score": judge_score,
            "llm_judge_reasoning": judge_reasoning,
            "ragas_faithfulness": ragas_scores.get("faithfulness"),
            "ragas_context_relevancy": ragas_scores.get("context_precision"),
            "ragas_context_recall": ragas_scores.get("context_recall"),
            "model_name": pipeline_res["model_name"]
        }
        
        insert_query = """
            INSERT INTO evaluation_results (
                question_id, question_text, ground_truth_answer_embedding,
                rag_answer_text, rag_answer_embedding, retrieved_context,
                cosine_similarity, model_name, llm_judge_score, llm_judge_reasoning,
                ragas_faithfulness, ragas_context_relevancy, ragas_context_recall
            ) VALUES (
                %(question_id)s, %(question_text)s, %(ground_truth_answer_embedding)s,
                %(rag_answer_text)s, %(rag_answer_embedding)s, %(retrieved_context)s,
                %(cosine_similarity)s, %(model_name)s, %(llm_judge_score)s, %(llm_judge_reasoning)s,
                %(ragas_faithfulness)s, %(ragas_context_relevancy)s, %(ragas_context_recall)s
            ) ON CONFLICT (question_id) DO NOTHING;
        """
        eval_cursor.execute(insert_query, final_result)
        rag_conn.commit()

    eval_cursor.close()
    print("   - Resultados salvos com sucesso no PostgreSQL!")
    print("\n--- Avaliação Completa do Graph RAG Concluída! ---")

if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    print("-> Carregando modelos e clientes (otimizados para custo)...")
    
    final_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    translator_llm = OpenAI(api_key=openai_api_key) 
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    scraping_conn = None
    neo4j_driver = None
    rag_conn = None
    
    try:
        print("-> Conectando ao PostgreSQL (Scrapping DB)...")
        scraping_conn = psycopg2.connect(
            dbname=os.getenv("SCRAP_PG_DATABASE"), user=os.getenv("SCRAP_PG_USER"),
            password=os.getenv("SCRAP_PG_PASSWORD"), host=os.getenv("SCRAP_PG_HOST"),
            port=os.getenv("SCRAP_PG_PORT")
        )
        scraping_cursor = scraping_conn.cursor(cursor_factory=RealDictCursor)

        print("-> Conectando ao Neo4j...")
        NEO4J_URI = os.getenv("NEO4J_URI")
        NEO4J_USER = os.getenv("NEO4J_USER")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()

        print("-> Conectando ao PostgreSQL (Graph RAG DB)...")
        rag_conn = psycopg2.connect(
            dbname=os.getenv("GRAPH_PG_DATABASE"),
            user=os.getenv("GRAPH_PG_USER"),
            password=os.getenv("GRAPH_PG_PASSWORD"),
            host=os.getenv("GRAPH_PG_HOST"),
            port=os.getenv("GRAPH_PG_PORT")
        )

        print("\n--- Conexões estabelecidas. Iniciando avaliação robusta e paralela. ---\n")
        run_evaluation(rag_conn, scraping_cursor, neo4j_driver, embeddings_model, translator_llm, final_llm, openai_api_key)

    except Exception as e:
        print(f"\nOcorreu um erro durante a avaliação: {e}")
        if rag_conn: rag_conn.rollback()

    finally:
        if scraping_conn:
            scraping_conn.close()
            print("\n-> Conexão com PostgreSQL (Scrapping) fechada.")
        if neo4j_driver:
            neo4j_driver.close()
            print("-> Conexão com Neo4j fechada.")
        if rag_conn:
            rag_conn.close()
            print("-> Conexão com PostgreSQL (Graph RAG DB) fechada.")