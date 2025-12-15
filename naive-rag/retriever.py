import numpy as np


def retrieve_context_from_db(user_query: str, db_cursor, embeddings_model, top_k: int = 5) -> str:
    """
    Busca no banco de dados os 'top_k' documentos mais relevantes para a pergunta do usuário.
    """
    print(f"-> Gerando embedding para a pergunta...")
    query_embedding = embeddings_model.embed_query(user_query)

    query_embedding_list = np.array(query_embedding).tolist()

    print(f"-> Buscando {top_k} documentos relevantes no banco de dados...")
    
    db_cursor.execute(
        """
        SELECT content FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (str(query_embedding_list), top_k)
    )
    
    results = db_cursor.fetchall()
    context = "\n\n".join([row[0] for row in results])
    
    print("-> Contexto recuperado com sucesso.")
    return context