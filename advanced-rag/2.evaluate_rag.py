import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from retriever import retrieve_context_from_db
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
    print("-> Buscando perguntas e respostas 'padrão ouro' do banco de scrapping...")
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
    tensor1 = torch.tensor(embedding1).unsqueeze(0)
    tensor2 = torch.tensor(embedding2).unsqueeze(0)
    return F.cosine_similarity(tensor1, tensor2).item()

def run_rag_pipeline_for_question(item: dict, db_params: dict, llm, embeddings_model) -> dict:
    question_id = item['id']
    question_text = item['pergunta']
    ground_truth_answer = item['ground_truth_answer']
    ground_truth_embedding = json.loads(item['ground_truth_embedding'])

    rag_conn_thread = None
    try:
        rag_conn_thread = psycopg2.connect(**db_params)
        rag_cursor_thread = rag_conn_thread.cursor()

        print(f"\n--- Processando Pergunta ID: {question_id} ---")
        print(f"   Pergunta: {question_text}")

        retrieved_context = retrieve_context_from_db(question_text, rag_cursor_thread, embeddings_model)
        rag_answer_text = generate_response(retrieved_context, question_text, llm)
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
            "model_name": llm.model_name
        }
    finally:
        if rag_conn_thread:
            rag_conn_thread.close()

def run_evaluation(rag_conn, rag_cursor, scraping_cursor, llm, embeddings_model, openai_api_key: str, db_params: dict):
    """Orquestra o processo de avaliação, incluindo a avaliação em lote com RAGAS."""
    questions_to_evaluate = fetch_questions_for_evaluation(scraping_cursor)
    rag_cursor.execute("SELECT question_id FROM evaluation_results")
    evaluated_ids = {row[0] for row in rag_cursor.fetchall()}
    questions_to_process = [item for item in questions_to_evaluate if item['id'] not in evaluated_ids]

    if not questions_to_process:
        print("-> Todas as perguntas já foram avaliadas. Nada a fazer.")
        return

    print(f"-> Fase 1: Executando o pipeline de RAG para {len(questions_to_process)} perguntas em paralelo...")
    
    pipeline_results = []
    MAX_WORKERS = 10
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(run_rag_pipeline_for_question, item, db_params, llm, embeddings_model): item
            for item in questions_to_process
        }
        for future in as_completed(future_to_question):
            try:
                result_data = future.result()
                pipeline_results.append(result_data)
                print(f"   - [ID {result_data['question_id']}] Pipeline RAG concluído.")
            except Exception as exc:
                question_item = future_to_question[future]
                print(f"   - ERRO ao processar pipeline da pergunta ID {question_item['id']}: {exc}")

    if not pipeline_results:
        print("\n--- Nenhuma pergunta foi processada com sucesso. Encerrando avaliação. ---")
        return

    print("\n\n-> Fase 2: Preparando e executando a avaliação em lote com RAGAS...")
    for r in pipeline_results:
        r['reference'] = r['ground_truth']
    ragas_dataset = Dataset.from_list(pipeline_results)
    
    metrics = [Faithfulness(), ContextPrecision(), ContextRecall()]
    
    ragas_result = evaluate(ragas_dataset, metrics=metrics)
    print("   - Avaliação com RAGAS concluída.")
    print(ragas_result)
    ragas_scores_df = ragas_result.to_pandas()

    print("\n-> Fase 3: Juntando resultados e salvando no banco de dados...")
    for i, pipeline_res in enumerate(pipeline_results):
        print(f"   - [ID {pipeline_res['question_id']}] Consolidando e salvando resultados...")
        judge_score, judge_reasoning = evaluate_with_llm_judge(
            question=pipeline_res["question"],
            ground_truth_answer=pipeline_res["ground_truth"],
            generated_answer=pipeline_res["answer"],
            api_key=openai_api_key
        )
        
        final_result_for_db = {
            "question_id": pipeline_res["question_id"],
            "question_text": pipeline_res["question"],
            "ground_truth_answer_embedding": pipeline_res["ground_truth_answer_embedding"],
            "rag_answer_text": pipeline_res["answer"],
            "rag_answer_embedding": pipeline_res["rag_answer_embedding"],
            "retrieved_context": "\n---\n".join(pipeline_res["contexts"]),
            "cosine_similarity": pipeline_res["cosine_similarity"],
            "model_name": pipeline_res["model_name"],
            "llm_judge_score": judge_score,
            "llm_judge_reasoning": judge_reasoning,
            "ragas_faithfulness": ragas_scores_df.iloc[i]["faithfulness"],
            "ragas_context_relevancy": ragas_scores_df.iloc[i]["context_precision"],
            "ragas_context_recall": ragas_scores_df.iloc[i]["context_recall"]
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
        rag_cursor.execute(insert_query, final_result_for_db)

    rag_conn.commit()
    print("   - Todos os novos resultados de avaliação foram salvos no banco de dados!")
    print("\n--- Avaliação Completa do RAG Avançado Concluída! ---")


if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    print("-> Carregando modelos (otimizados para custo)...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    rag_conn = None
    scraping_conn = None
    try:
        adv_db_params = {
            "dbname": os.getenv("ADV_PG_DATABASE"),
            "user": os.getenv("ADV_PG_USER"),
            "password": os.getenv("ADV_PG_PASSWORD"),
            "host": os.getenv("ADV_PG_HOST"),
            "port": os.getenv("ADV_PG_PORT")
        }
        rag_conn = psycopg2.connect(**adv_db_params)
        rag_cursor = rag_conn.cursor()

        scraping_conn = psycopg2.connect(
            dbname=os.getenv("SCRAP_PG_DATABASE"), user=os.getenv("SCRAP_PG_USER"),
            password=os.getenv("SCRAP_PG_PASSWORD"), host=os.getenv("SCRAP_PG_HOST"),
            port=os.getenv("SCRAP_PG_PORT")
        )
        scraping_cursor = scraping_conn.cursor(cursor_factory=RealDictCursor)

        print("\n--- Conexões estabelecidas. Iniciando avaliação robusta e paralela. ---\n")
        run_evaluation(rag_conn, rag_cursor, scraping_cursor, llm, embeddings_model, openai_api_key, adv_db_params)

    except Exception as e:
        print(f"\nOcorreu um erro durante a avaliação: {e}")
        if rag_conn: rag_conn.rollback()

    finally:
        if rag_conn: rag_conn.close()
        if scraping_conn: scraping_conn.close()