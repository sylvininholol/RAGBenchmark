import numpy as np
from sentence_transformers import CrossEncoder

def search_summaries_for_documents(user_query: str, db_cursor, embeddings_model, top_k: int = 3) -> list[int]:
    """
    Etapa 1: Busca nos resumos dos documentos para encontrar os DOCUMENTOS mais relevantes.
    Retorna uma lista de IDs de documentos.
    """
    print(f"-> Etapa 1: Buscando {top_k} resumos de documentos mais relevantes...")
    query_embedding = embeddings_model.embed_query(user_query)
    query_embedding_list = np.array(query_embedding).tolist()

    db_cursor.execute(
        """
        SELECT document_id FROM document_summaries
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (str(query_embedding_list), top_k)
    )

    results = db_cursor.fetchall()
    if not results:
        print("   - Nenhum documento relevante encontrado na busca por resumos.")
        return []

    document_ids = [row[0] for row in results]
    print(f"   - Documentos relevantes encontrados (IDs): {document_ids}")
    return document_ids

def retrieve_chunks_from_documents(document_ids: list[int], db_cursor, chunks_per_doc: int = 10) -> list[tuple]:
    """
    Etapa 2: Recupera os 'chunks_per_doc' chunks mais relevantes de uma lista específica de documentos.
    A busca ainda é baseada na similaridade vetorial para um pré-filtro.
    """
    if not document_ids:
        return []

    print(f"-> Etapa 2: Recuperando até {chunks_per_doc} chunks dos documentos selecionados...")

    db_cursor.execute(
        """
        SELECT id, content FROM chunks
        WHERE page_id IN (
            SELECT id FROM pages WHERE document_id = ANY(%s)
        )
        LIMIT %s
        """,
        (document_ids, len(document_ids) * chunks_per_doc)
    )

    chunks = db_cursor.fetchall()
    print(f"   - {len(chunks)} chunks recuperados para re-ranking.")
    return chunks

def rerank_chunks(user_query: str, chunks: list[tuple], top_n: int = 5) -> list[str]:
    """
    Etapa 3: Re-rankeia os chunks recuperados usando um modelo Cross-Encoder mais preciso.
    """
    if not chunks:
        return []

    print(f"-> Etapa 3: Re-rankeando {len(chunks)} chunks com Cross-Encoder...")

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

    chunk_contents = [chunk[1] for chunk in chunks]
    pairs = [(user_query, content) for content in chunk_contents]

    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(scores, chunk_contents))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    reranked_chunks = [content for score, content in scored_chunks[:top_n]]

    print(f"   - Re-ranking concluído. Selecionando os {top_n} melhores chunks.")
    return reranked_chunks

def retrieve_context_from_db(user_query: str, db_cursor, embeddings_model, top_k: int = 5) -> str:
    """
    Orquestra o pipeline completo de recuperação avançada:
    1. Busca por resumos para achar documentos.
    2. Recupera chunks desses documentos.
    3. Re-rankeia os chunks para obter o contexto mais relevante.
    """
    print("\n--- Iniciando Pipeline de Recuperação Avançada ---")

    relevant_doc_ids = search_summaries_for_documents(user_query, db_cursor, embeddings_model, top_k=3)
    if not relevant_doc_ids:
        return "Não foi possível encontrar documentos relevantes para a sua pergunta."

    initial_chunks = retrieve_chunks_from_documents(relevant_doc_ids, db_cursor, chunks_per_doc=8)
    if not initial_chunks:
        return "Documentos relevantes foram encontrados, mas não foi possível recuperar seus conteúdos (chunks)."

    final_chunks = rerank_chunks(user_query, initial_chunks, top_n=top_k)

    context = "\n\n---\n\n".join(final_chunks)

    print("--- Contexto final recuperado e re-rankeado com sucesso! ---")
    return context