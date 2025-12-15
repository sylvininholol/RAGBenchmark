import json
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from openai import OpenAI

def extract_entities_from_query(query: str, llm_client: OpenAI) -> list[str]:
    """
    Usa um LLM leve para extrair entidades nomeadas (Pessoa, Organização, Local)
    da pergunta do usuário.
    """
    print(f"-> Extraindo entidades da pergunta: '{query}'")
    prompt = f"""
    Analise a pergunta do usuário e extraia os nomes próprios de pessoas, organizações (times, clubes), ou locais.
    Retorne um objeto JSON com uma única chave "entities", contendo uma lista de strings com os nomes.
    Se nenhuma entidade for encontrada, retorne {{"entities": []}}.

    Exemplo 1:
    Pergunta: "Qual foi o placar do jogo do Botafogo contra o Palmeiras?"
    Saída: {{"entities": ["Botafogo", "Palmeiras"]}}
    
    Exemplo 2:
    Pergunta: "O que aconteceu com Gabigol?"
    Saída: {{"entities": ["Gabigol"]}}

    Exemplo 3:
    Pergunta: "Como foi a temporada?"
    Saída: {{"entities": []}}

    Pergunta para análise:
    "{query}"
    """
    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": 'Você extrai entidades de uma pergunta e retorna um JSON com a chave "entities".'},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        entities = result.get("entities", [])
        if entities:
            print(f"   - Entidades extraídas: {entities}")
        return entities
    except Exception as e:
        print(f"ERRO: Falha ao extrair entidades da pergunta: {e}")
        return []

def retrieve_graph_neighborhood(embedding: list, 
                                entities: list[str],
                                driver: GraphDatabase.driver, 
                                top_k: int = 5) -> str: 
    """
    Busca os nós :Chunk mais relevantes (híbrido), faz reranking
    e expande o contexto de forma LIMITADA.
    """
    print(f"-> Buscando chunks (vetor/entidade) e fazendo reranking para top_k={top_k}...")
    contexts = []
    
    vector_chunk_scores = {} 
    entity_chunk_ids = set()

    with driver.session(database="neo4j") as session:
        try:
            similar_chunks_result = session.run(
                """
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k_vector, $embedding) 
                YIELD node, score 
                RETURN elementId(node) AS id, score
                """,
                top_k_vector=top_k, embedding=embedding 
            )
            vector_chunk_scores = {record['id']: record['score'] for record in similar_chunks_result}
            print(f"   - Busca vetorial encontrou {len(vector_chunk_scores)} chunks.")
        except Exception as e:
            print(f"   - AVISO: Busca vetorial falhou: {e}")

        if entities:
            try:
                entity_chunks_result = session.run(
                    """
                    UNWIND $entities AS entity_name
                    MATCH (e:Entidade) WHERE toLower(e.nome) = toLower(entity_name)
                    MATCH (e)<-[:MENCIONA]-(c:Chunk)
                    RETURN elementId(c) AS id
                    """,
                    entities=entities
                )
                entity_chunk_ids = {record['id'] for record in entity_chunks_result}
                print(f"   - Busca por entidade encontrou {len(entity_chunk_ids)} chunks.")
            except Exception as e:
                print(f"   - AVISO: Busca por entidade falhou: {e}")

        all_chunk_ids = set(vector_chunk_scores.keys()) | entity_chunk_ids
        
        scored_chunk_ids = []
        for chunk_id in all_chunk_ids:
            score = vector_chunk_scores.get(chunk_id, 0.0)
            if chunk_id in entity_chunk_ids:
                score += 1.0 
            scored_chunk_ids.append((chunk_id, score))
        
        scored_chunk_ids.sort(key=lambda x: x[1], reverse=True)
        
        top_chunk_ids = [item[0] for item in scored_chunk_ids[:top_k]]
        
        if not top_chunk_ids:
            return "Nenhum contexto relevante encontrado no grafo."

        print(f"   - Total de {len(all_chunk_ids)} chunks únicos encontrados. Usando os {len(top_chunk_ids)} mais relevantes após reranking.")

        for chunk_id in top_chunk_ids:
            result = session.run(
                """
                MATCH (c:Chunk) WHERE elementId(c) = $chunk_id
                
                OPTIONAL MATCH (prev_chunk)-[:PROXIMO_CHUNK]->(c)
                OPTIONAL MATCH (c)-[:PROXIMO_CHUNK]->(next_chunk)
                OPTIONAL MATCH (c)-[:PARTE_DE]->(d:Documento)
                
                OPTIONAL MATCH (c)-[:MENCIONA]->(entity:Entidade)
                WITH c, d, prev_chunk, next_chunk, COLLECT(DISTINCT entity) AS mentioned_entities

                UNWIND CASE WHEN size(mentioned_entities) = 0 THEN [null] ELSE mentioned_entities END AS entity

                // --- ALTERAÇÃO PRINCIPAL AQUI ---
                // Adicionamos um 'WITH ... LIMIT 5' para controlar a explosão de contexto
                CALL {
                    WITH entity
                    MATCH (entity)-[r]-(neighbor:Entidade)
                    // Limita a 5 relacionamentos POR entidade mencionada
                    WITH entity, r, neighbor LIMIT 5 
                    RETURN COLLECT(
                        "(" + coalesce(entity.nome, 'Entidade') + ") " +
                        "-[:" + type(r) + " " + apoc.convert.toJson(properties(r)) + "]-> " +
                        "(" + coalesce(neighbor.nome, 'Entidade') + ")"
                    ) AS relationships
                }
                // --- FIM DA ALTERAÇÃO ---
                
                RETURN
                    c.text AS chunk_text,
                    d.titulo AS document_title,
                    prev_chunk.text AS prev_chunk_text,
                    next_chunk.text AS next_chunk_text,
                    [ent IN mentioned_entities | ent.nome] AS entity_names,
                    COLLECT(relationships) AS all_relationships_nested
                """,
                chunk_id=chunk_id
            ).single()

            if result:
                doc_title = result.get("document_title", "Fonte desconhecida")
                prev_text = result.get("prev_chunk_text", "")
                chunk_text = result.get("chunk_text", "")
                next_text = result.get("next_chunk_text", "")
                entity_names = result.get("entity_names", [])
                
                all_relationships_nested = result.get("all_relationships_nested", [])
                all_relationships = [item for sublist in all_relationships_nested for item in sublist if item]

                full_chunk_context = f"...{prev_text} {chunk_text} {next_text}..."

                context_str = f"Fonte: {doc_title}\n"
                context_str += f"Contexto do Documento: \"{full_chunk_context}\"\n"
                if entity_names:
                    context_str += f"Entidades Mencionadas no Trecho: {', '.join(entity_names)}\n"
                if all_relationships:
                    unique_rels = sorted(list(set(all_relationships)))
                    context_str += "Fatos Relacionados no Grafo:\n" + "\n".join([f"- {rel}" for rel in unique_rels])
                
                contexts.append(context_str)
    
    return "\n\n---\n\n".join(contexts) if contexts else "Nenhum contexto detalhado encontrado no grafo."

def translate_graph_to_natural_language(context: str, llm_client: OpenAI) -> str:
    
    """Usa um LLM para converter o contexto estruturado do grafo em texto fluído."""
    prompt = f"""Você é um assistente especialista em transformar dados estruturados de um grafo de conhecimento 
                    em uma descrição textual concisa e informativa. Seu objetivo é gerar uma versão em 
                    linguagem natural do contexto fornecido, mantendo todas as informações cruciais, mas de forma
                  compacta, ideal para ser usada como fonte de conhecimento para outro LLM em um sistema RAG.

    Contexto Estruturado do Grafo:
    ---
    {context}
    ---

    Produza APENAS a tradução em linguagem natural compacta.
    """
    print("-> Traduzindo contexto do grafo para linguagem natural usando LLM...")
    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERRO: Falha ao chamar LLM para tradução: {e}")
        return "Erro na tradução do contexto."

def retrieve_context_from_graph(user_query: str, 
                                neo4j_driver: GraphDatabase.driver, 
                                embeddings_model: OpenAIEmbeddings, 
                                llm_translator_client: OpenAI) -> str:
    """
    Orquestra o processo completo de recuperação de contexto do Graph RAG.
    (HÍBRIDO: VETOR + ENTIDADE)
    """
    print(f"-> Gerando embedding para a pergunta: '{user_query}'")
    try:
        query_embedding = embeddings_model.embed_query(user_query)
    except Exception as e:
        print(f"ERRO: Falha ao gerar embedding para a pergunta: {e}")
        return "Erro ao processar a pergunta."

    extracted_entities = extract_entities_from_query(user_query, llm_translator_client)

    structured_context = retrieve_graph_neighborhood(
        query_embedding, 
        extracted_entities, 
        neo4j_driver,
        top_k=3
    )

    if "Nenhum" in structured_context or "Erro" in structured_context:
        return structured_context

    natural_language_context = translate_graph_to_natural_language(structured_context, llm_translator_client)
    
    return natural_language_context