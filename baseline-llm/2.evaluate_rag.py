import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from generate_response import generate_response
from llm_judge import evaluate_with_llm_judge

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
    tensor1 = torch.tensor(embedding1).unsqueeze(0)
    tensor2 = torch.tensor(embedding2).unsqueeze(0)
    return F.cosine_similarity(tensor1, tensor2).item()

def run_baseline_pipeline_for_question(item: dict, llm, embeddings_model) -> dict:
    """
    Executa o pipeline do LLM baseline para uma única pergunta
    e retorna todos os artefatos necessários para a avaliação.
    """
    question_id = item['id']
    question_text = item['pergunta']
    ground_truth_answer = item['ground_truth_answer']
    ground_truth_embedding = json.loads(item['ground_truth_embedding'])

    print(f"\n--- Processando Pergunta ID: {question_id} ---")
    print(f"   Pergunta: {question_text}")

    baseline_answer_text = generate_response(question_text, llm)
    baseline_answer_embedding = embeddings_model.embed_query(baseline_answer_text)
    similarity_score = calculate_cosine_similarity(ground_truth_embedding, baseline_answer_embedding)
    print(f"   - [ID {question_id}] Similaridade de Cosseno: {similarity_score:.4f}")

    return {
        "question_id": question_id,
        "question": question_text,
        "answer": baseline_answer_text,
        "ground_truth": ground_truth_answer,
        "baseline_answer_embedding": str(baseline_answer_embedding),
        "ground_truth_answer_embedding": str(ground_truth_embedding),
        "cosine_similarity": similarity_score,
        "model_name": llm.model_name
    }

def run_evaluation(baseline_conn, baseline_cursor, scraping_cursor, llm, embeddings_model, openai_api_key: str):
    """Orquestra o processo de avaliação do LLM baseline."""
    questions_to_evaluate = fetch_questions_for_evaluation(scraping_cursor)

    baseline_cursor.execute("SELECT question_id FROM evaluation_results")
    evaluated_ids = {row[0] for row in baseline_cursor.fetchall()}
    questions_to_process = [item for item in questions_to_evaluate if item['id'] not in evaluated_ids]

    if not questions_to_process:
        print("-> Todas as perguntas já foram avaliadas. Nada a fazer.")
        return

    print(f"-> Fase 1: Executando o pipeline do LLM baseline para {len(questions_to_process)} perguntas em paralelo...")
    
    pipeline_results = []
    MAX_WORKERS = 10

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(run_baseline_pipeline_for_question, item, llm, embeddings_model): item
            for item in questions_to_process
        }
        for future in as_completed(future_to_question):
            try:
                result_data = future.result()
                pipeline_results.append(result_data)
                print(f"   - [ID {result_data['question_id']}] Pipeline concluído.")
            except Exception as exc:
                question_item = future_to_question[future]
                print(f"   - ERRO ao processar pipeline da pergunta ID {question_item['id']}: {exc}")

    if not pipeline_results:
        print("\n--- Nenhuma pergunta foi processada com sucesso. Encerrando avaliação. ---")
        return

    print("\n-> Fase 2: Juntando resultados e salvando no banco de dados...")
    
    for res in pipeline_results:
        print(f"   - [ID {res['question_id']}] Consolidando e salvando resultados...")
        
        judge_score, judge_reasoning = evaluate_with_llm_judge(
            question=res["question"],
            ground_truth_answer=res["ground_truth"],
            generated_answer=res["answer"],
            api_key=openai_api_key
        )
        
        final_result_for_db = {
            "question_id": res["question_id"],
            "question_text": res["question"],
            "ground_truth_answer_embedding": res["ground_truth_answer_embedding"],
            "baseline_answer_text": res["answer"],
            "baseline_answer_embedding": res["baseline_answer_embedding"],
            "cosine_similarity": res["cosine_similarity"],
            "model_name": res["model_name"],
            "llm_judge_score": judge_score,
            "llm_judge_reasoning": judge_reasoning,
        }
        
        insert_query = """
            INSERT INTO evaluation_results (
                question_id, question_text, ground_truth_answer_embedding,
                baseline_answer_text, baseline_answer_embedding,
                cosine_similarity, model_name, llm_judge_score, llm_judge_reasoning
            ) VALUES (
                %(question_id)s, %(question_text)s, %(ground_truth_answer_embedding)s,
                %(baseline_answer_text)s, %(baseline_answer_embedding)s,
                %(cosine_similarity)s, %(model_name)s, %(llm_judge_score)s, %(llm_judge_reasoning)s
            ) ON CONFLICT (question_id) DO NOTHING;
        """
        baseline_cursor.execute(insert_query, final_result_for_db)

    baseline_conn.commit()
    print("   - Todos os novos resultados de avaliação foram salvos no banco de dados!")
    print("\n--- Avaliação Completa do LLM Baseline Concluída! ---")

if __name__ == "__main__":
    print("--- Iniciando Script de Avaliação do LLM Baseline ---")
    load_dotenv()

    print("-> Carregando modelos...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    baseline_conn = None
    scraping_conn = None
    try:
        print("-> Conectando ao banco de dados do Baseline (baseline_db)...")
        baseline_conn = psycopg2.connect(
            dbname=os.getenv("BASELINE_PG_DATABASE"), user=os.getenv("BASELINE_PG_USER"),
            password=os.getenv("BASELINE_PG_PASSWORD"), host=os.getenv("BASELINE_PG_HOST"),
            port=os.getenv("BASELINE_PG_PORT")
        )
        baseline_cursor = baseline_conn.cursor()

        print("-> Conectando ao banco de dados do Web Scrapping (wscrap_db)...")
        scraping_conn = psycopg2.connect(
            dbname=os.getenv("SCRAP_PG_DATABASE"), user=os.getenv("SCRAP_PG_USER"),
            password=os.getenv("SCRAP_PG_PASSWORD"), host=os.getenv("SCRAP_PG_HOST"),
            port=os.getenv("SCRAP_PG_PORT")
        )
        scraping_cursor = scraping_conn.cursor(cursor_factory=RealDictCursor)

        print("\n--- Conexões estabelecidas. Iniciando avaliação. ---\n")
        run_evaluation(baseline_conn, baseline_cursor, scraping_cursor, llm, embeddings_model, openai_api_key)

    except Exception as e:
        print(f"\nOcorreu um erro durante a avaliação: {e}")
        if baseline_conn: baseline_conn.rollback()

    finally:
        if baseline_conn: baseline_conn.close()
        if scraping_conn: scraping_conn.close()