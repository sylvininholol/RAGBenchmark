import os
import psycopg2
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

from generate_response import generate_response
from retriever import retrieve_context_from_db


if __name__ == "__main__":

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print("-> Carregando modelos...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    conn = None
    try:
        print("-> Conectando ao banco de dados PostgreSQL...")
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"), user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"), host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cursor = conn.cursor()


        print("--- Configuração Concluída ---")
        query = "Qual é a experiência profissional do Gabriel Marinho mencionada no documento?"
        
        # Rag de fato, onde está acontecendo o retrieval
        retrieved_context = retrieve_context_from_db(query, cursor, embeddings_model)

        final_response = generate_response(retrieved_context, query, llm)
        print(f"\n-------------- Resposta Final -------------\n\n{final_response}")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    
    finally:
        if conn:
            print("\n-> Fechando conexão com o banco de dados.")
            conn.close()