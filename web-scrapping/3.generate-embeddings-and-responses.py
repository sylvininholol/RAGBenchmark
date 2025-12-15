import os
import psycopg2
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

DB_CONFIG = {
    'host': os.getenv("PG_HOST"),
    'port': os.getenv("PG_PORT"),
    'user': os.getenv("PG_USER"),
    'password': os.getenv("PG_PASSWORD"),
    'dbname': os.getenv("PG_DATABASE")
}

client = OpenAI(api_key=openai_api_key)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

ANSWER_PROMPT = (
    "Com base no texto fornecido, responda à seguinte pergunta de forma clara e precisa:\n\n"
    "Texto:\n{texto}\n\n"
    "Pergunta:\n{pergunta}\n\n"
    "Resposta:"
)

def gerar_resposta(texto, pergunta):
    prompt = ANSWER_PROMPT.format(texto=texto, pergunta=pergunta)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        resposta = response.choices[0].message.content.strip()
        return resposta
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return None

def gerar_embedding(texto):
    try:
        embedding = embeddings_model.embed_query(texto)
        return embedding
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # 1. Gerar embeddings para perguntas que ainda não têm
    print("Gerando embeddings para perguntas...")
    cur.execute("""
        SELECT p.id, p.pergunta 
        FROM perguntas p 
        WHERE p.embedding IS NULL
    """)
    perguntas_sem_embedding = cur.fetchall()
    
    for pergunta_id, pergunta in perguntas_sem_embedding:
        print(f"Gerando embedding para pergunta id {pergunta_id}...")
        embedding = gerar_embedding(pergunta)
        if embedding:
            cur.execute("UPDATE perguntas SET embedding = %s WHERE id = %s", (embedding, pergunta_id))
            conn.commit()
            print(f"Embedding salvo para pergunta id {pergunta_id}.")
    
    # 2. Gerar respostas para perguntas que ainda não têm resposta na tabela respostas
    print("\nGerando respostas para perguntas...")
    cur.execute("""
        SELECT p.id, p.pergunta, s.texto 
        FROM perguntas p 
        JOIN scraping s ON p.scraping_id = s.id 
        WHERE NOT EXISTS (
            SELECT 1 FROM respostas r WHERE r.pergunta_id = p.id
        )
    """)
    perguntas_sem_resposta = cur.fetchall()
    
    for pergunta_id, pergunta, texto in perguntas_sem_resposta:
        print(f"Gerando resposta para pergunta id {pergunta_id}...")
        resposta = gerar_resposta(texto, pergunta)
        if resposta:
            embedding_resposta = gerar_embedding(resposta)
            cur.execute(
                "INSERT INTO respostas (pergunta_id, resposta, embedding) VALUES (%s, %s, %s)",
                (pergunta_id, resposta, embedding_resposta)
            )
            conn.commit()
            print(f"Resposta inserida na tabela respostas para pergunta id {pergunta_id}.")
    
    cur.close()
    conn.close()
    print("\nProcesso concluído!")

if __name__ == "__main__":
    main() 