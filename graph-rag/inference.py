import os
from neo4j import GraphDatabase, basic_auth
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv

from retriever import retrieve_context_from_graph
from generate_response import generate_response

if __name__ == "__main__":
    load_dotenv()

    print("-> Carregando modelos e clientes...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    final_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    translator_llm = OpenAI(api_key=openai_api_key)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    neo4j_driver = None
    try:
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print(f"-> Conectado ao Neo4j em {NEO4J_URI}")
        print("--- Configuração Concluída ---")

        while True:
            query = input("\nDigite sua pergunta (ou 'sair' para terminar): ")
            if query.lower() == 'sair':
                break

            retrieved_context = retrieve_context_from_graph(query, neo4j_driver, embeddings_model, translator_llm)

            print("\n--- CONTEXTO FINAL (EM LINGUAGEM NATURAL) PARA O RAG ---")
            print(retrieved_context)
            print("------------------------------------------------------")

            final_response = generate_response(retrieved_context, query, final_llm)

            print(f"\n-------------- Resposta Final -------------\n")
            print(final_response)

    except Exception as e:
        print(f"Ocorreu um erro no pipeline principal: {e}")

    finally:
        if neo4j_driver:
            print("\n-> Fechando conexão com o Neo4j.")
            neo4j_driver.close()