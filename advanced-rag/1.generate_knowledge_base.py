import os
from pathlib import Path
import psycopg2
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np


def get_or_create_document(cursor, title: str) -> int:
    """Verifica se um documento com o mesmo título já existe para evitar duplicatas."""
    print(f"-> Verificando/Criando registro para o documento: {title}")
    cursor.execute("SELECT id FROM documents WHERE title = %s", (title,))
    result = cursor.fetchone()
    if result:
        print(f"   - Documento '{title}' já existe. Pulando inserção de conteúdo.")
        return result[0], True
    else:
        cursor.execute("INSERT INTO documents (title) VALUES (%s) RETURNING id", (title,))
        doc_id = cursor.fetchone()[0]
        print(f"   - Documento '{title}' criado com ID: {doc_id}.")
        return doc_id, False

def store_content_as_page(cursor, document_id: int, content: str) -> int:
    """
    Armazena um bloco de conteúdo (como o texto de uma notícia) como uma única "página".
    Isso adapta o schema existente (document->page->chunk) para fontes não paginadas.
    """
    print(f"   - Inserindo conteúdo como página 1 para o documento ID {document_id}...")
    cursor.execute(
        "INSERT INTO pages (document_id, page_number, content) VALUES (%s, %s, %s) RETURNING id",
        (document_id, 1, content)
    )
    return cursor.fetchone()[0]

def split_and_store_chunks(cursor, page_id: int, page_content: str):
    """Divide o conteúdo da página em chunks e os armazena, evitando duplicatas."""
    print("-> Dividindo conteúdo em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks_content = text_splitter.split_text(page_content)
    for i, chunk_content in enumerate(chunks_content, start=1):
        print(f"   - Inserindo chunk {i} da página ID {page_id}...")
        cursor.execute(
            "INSERT INTO chunks (page_id, chunk_number, content) VALUES (%s, %s, %s)",
            (page_id, i, chunk_content)
        )

def generate_and_store_embeddings(cursor, embeddings_model):
    """Gera embeddings para todos os chunks que ainda não os possuem."""
    print("-> Verificando chunks que precisam de embedding...")
    cursor.execute("SELECT id, content FROM chunks WHERE embedding IS NULL")
    chunks_to_embed = cursor.fetchall()

    if not chunks_to_embed:
        print("-> Nenhum chunk novo para gerar embedding. Base de dados atualizada.")
        return

    print(f"-> Gerando embeddings para {len(chunks_to_embed)} chunks novos...")
    contents_to_embed = [chunk[1] for chunk in chunks_to_embed]
    embeddings = embeddings_model.embed_documents(contents_to_embed)

    for i, (chunk_id, _) in enumerate(chunks_to_embed):
        embedding_list = np.array(embeddings[i]).tolist()
        cursor.execute("UPDATE chunks SET embedding = %s WHERE id = %s", (str(embedding_list), chunk_id))
    print("-> Embeddings gerados e armazenados com sucesso.")

def fetch_news_from_scraping_db(scraping_cursor):
    """Busca todas as notícias do banco de dados de web-scrapping."""
    print("-> Buscando notícias do banco de dados de scrapping...")
    scraping_cursor.execute("SELECT titulo, texto FROM scraping WHERE texto IS NOT NULL AND titulo IS NOT NULL")
    news = scraping_cursor.fetchall()
    print(f"   - {len(news)} notícias encontradas.")
    return news

def ingest_scraped_data_to_rag_db(rag_conn, rag_cursor, scraping_cursor, embeddings_model):
    """Orquestra o processo completo de ingestão dos dados de scrapping."""

    news_to_process = fetch_news_from_scraping_db(scraping_cursor)

    for title, content in news_to_process:
        document_id, already_exists = get_or_create_document(rag_cursor, title)

        if already_exists:
            continue

        page_id = store_content_as_page(rag_cursor, document_id, content)

        split_and_store_chunks(rag_cursor, page_id, content)

        print(f"-> Comitando inserções de texto para '{title}'...")
        rag_conn.commit()

    generate_and_store_embeddings(rag_cursor, embeddings_model)

    print("-> Comitando embeddings no banco de dados...")
    rag_conn.commit()
    print("\n--- Processo de ingestão dos dados de scrapping concluído com sucesso! ---")

def generate_and_store_summaries(rag_conn, rag_cursor, embeddings_model):
    """
    Gera resumos para cada documento que ainda não tem um, e armazena no DB.
    """
    print("\n--- Iniciando Geração de Resumos de Documentos ---")

    rag_cursor.execute("""
        SELECT d.id, p.content
        FROM documents d
        JOIN pages p ON d.id = p.document_id
        LEFT JOIN document_summaries ds ON d.id = ds.document_id
        WHERE ds.id IS NULL
    """)
    docs_to_summarize = rag_cursor.fetchall()

    if not docs_to_summarize:
        print("-> Todos os documentos já possuem resumos. Nada a fazer.")
        return

    print(f"-> Encontrados {len(docs_to_summarize)} documentos para resumir...")

    # Usando LLM para summarizer
    from langchain_openai import ChatOpenAI
    summarizer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    summary_prompt = ChatPromptTemplate.from_template(
        "Sua tarefa é criar um resumo conciso e denso em informações do texto a seguir. "
        "O resumo deve capturar as entidades chave, seus atributos e relacionamentos principais. "
        "Este resumo será usado por uma IA para decidir se o documento completo é relevante para uma pergunta. "
        "Foque em nomes, locais, eventos, datas e conclusões importantes.\n\n"
        "Texto:\n---\n{document_content}\n---\n\nResumo Conciso:"
    )

    summarize_chain = summary_prompt | summarizer_llm

    for doc_id, content in docs_to_summarize:
        print(f"   - Gerando resumo para o documento ID: {doc_id}")

        response = summarize_chain.invoke({"document_content": content})
        summary_text = response.content

        summary_embedding = embeddings_model.embed_query(summary_text)

        rag_cursor.execute(
            """
            INSERT INTO document_summaries (document_id, summary_text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE
            SET summary_text = EXCLUDED.summary_text, embedding = EXCLUDED.embedding;
            """,
            (doc_id, summary_text, str(summary_embedding))
        )
        print(f"   - Resumo para o documento ID {doc_id} armazenado com sucesso.")

    rag_conn.commit()
    print("--- Geração de Resumos Concluída ---")

if __name__ == "__main__":
    print("--- Iniciando Processo de Ingestão de Documentos ---")
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    rag_conn = None
    scraping_conn = None
    try:
        print("-> Conectando ao banco de dados do Advanced RAG (adv_rag_db)...")
        rag_conn = psycopg2.connect(
            dbname=os.getenv("ADV_PG_DATABASE"),
            user=os.getenv("ADV_PG_USER"),
            password=os.getenv("ADV_PG_PASSWORD"),
            host=os.getenv("ADV_PG_HOST"),
            port=os.getenv("ADV_PG_PORT")
        )
        rag_cursor = rag_conn.cursor()

        print("-> Conectando ao banco de dados do Web Scrapping (wscrap_db)...")
        scraping_conn = psycopg2.connect(
            dbname=os.getenv("SCRAP_PG_DATABASE"), user=os.getenv("SCRAP_PG_USER"),
            password=os.getenv("SCRAP_PG_PASSWORD"), host=os.getenv("SCRAP_PG_HOST"),
            port=os.getenv("SCRAP_PG_PORT")
        )
        scraping_cursor = scraping_conn.cursor()

        print("\n--- Conexões estabelecidas. Iniciando ingestão. ---\n")

        ingest_scraped_data_to_rag_db(rag_conn, rag_cursor, scraping_cursor, embeddings_model)

        generate_and_store_summaries(rag_conn, rag_cursor, embeddings_model)

    except Exception as e:
        print(f"\nOcorreu um erro durante a ingestão: {e}")
        if rag_conn:
            rag_conn.rollback()

    finally:
        if rag_conn:
            print("-> Fechando conexão com o banco de dados do RAG.")
            rag_conn.close()
        if scraping_conn:
            print("-> Fechando conexão com o banco de dados do Web Scrapping.")
            scraping_conn.close()