import json
import os
import time
import psycopg2
from neo4j import GraphDatabase, basic_auth
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


def extract_graph_from_text(text_input: str, llm_client: OpenAI) -> dict:
    """Usa um LLM para extrair uma estrutura de grafo de conhecimento (entidades e relacionamentos) de um texto."""
    schema = """
    {
      "node_labels": ["Pessoa", "Organizacao", "Localizacao", "Evento", "Competicao", "Posicao", "Conceito"],
      "relationship_types": [
        "TRABALHA_EM", "JOGA_EM", "TREINA", "CONVOCADO_PARA", "TRANSFERIDO_PARA", "EMPRESTADO_POR",
        "PARTICIPOU_DE", "COMPETIU_EM", "VENCEU", "PERDEU_PARA", "CAMPEAO_DE", "VICE_CAMPEAO_DE",
        "MARCOU_GOL_EM", "DEU_ASSISTENCIA_PARA", "SOFREU_FALTA_DE", "LOCALIZADO_EM", "SEDE_DE",
        "PARTE_DE", "E_COMPANHEIRO_DE_EQUIPE_DE", "E_RIVAL_DE", "E_IDOLO_DE", "JOGA_NA_POSICAO"
      ]
    }
    """
    prompt = f"""Sua tarefa é analisar o texto de uma notícia esportiva e extrair uma rede de conhecimento em formato JSON, aderindo ESTRITAMENTE ao schema fornecido.

    **Schema Obrigatório:**
    {schema}

    **Instruções Detalhadas:**
    1.  Identifique entidades chave (pessoas, times, campeonatos, etc.) e atribua a elas um "label" OBRIGATORIAMENTE da lista "node_labels".
    2.  Para cada nó de entidade, o JSON DEVE conter "id" (um nome único temporário), "label" e "properties" (com a propriedade "nome" obrigatória).
    3.  Identifique relacionamentos de domínio entre essas entidades. O campo "type" DEVE OBRIGATORIAMENTE ser um dos valores da lista "relationship_types". *É PROIBIDO criar novos tipos de relacionamento*.
    4.  **Regra de Mapeamento Semântico:** Se um fato no texto não corresponder exatamente a um tipo de relacionamento do schema, sua tarefa é generalizar o fato e escolher o tipo mais próximo e semanticamente apropriado da lista, sem criar novos relacionamentos.
    5.  Para cada relacionamento, o JSON DEVE conter "source_id", "target_id", "type", e "properties". Se o texto mencionar detalhes sobre a relação (ex: placar de um jogo, valor de uma transferência), adicione-os como pares chave-valor dentro de "properties".
    6.  Seja o mais denso possível, extraindo todos os fatos e conexões relevantes que se conformem ao schema.

    **Exemplo do Formato de Saída OBRIGATÓRIO:**
    {{"nodes": [{{"id": "temp_pessoa_1", "label": "Pessoa", "properties": {{"nome": "Gabigol"}}}}, {{"id": "temp_org_1", "label": "Organizacao", "properties": {{"nome": "Flamengo"}}}}], "relationships": [{{"source_id": "temp_pessoa_1", "target_id": "temp_org_1", "type": "JOGA_EM", "properties": {{"ano_contrato": 2019}}}} ]}}

    **Texto para Análise:**
    "{text_input}"
    """
    print("-> Enviando texto para o LLM para extração de entidades e relações de domínio...")
    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um assistente especialista em extrair grafos de conhecimento de textos esportivos, seguindo um schema JSON rigoroso e obrigatório."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"ERRO: Ocorreu um erro ao chamar o LLM para extração: {e}")
        return {"nodes": [], "relationships": []}

def add_embeddings_to_nodes(graph_data: dict, embeddings_model: OpenAIEmbeddings) -> dict:
    """Gera e adiciona embeddings vetoriais às propriedades dos nós no grafo."""
    nodes = graph_data.get("nodes", [])
    if not nodes:
        return graph_data

    print(f"-> Gerando embeddings para {len(nodes)} nós extraídos...")

    texts_to_embed = [f"{node.get('label', 'Entidade')}: {node.get('properties', {}).get('nome', '')}" for node in nodes]

    valid_texts_with_indices = [
        (i, text) for i, text in enumerate(texts_to_embed)
        if nodes[i].get("properties", {}).get("nome")
    ]

    if not valid_texts_with_indices:
        print("   - Nenhum nó com conteúdo válido para gerar embeddings.")
        return graph_data

    indices, texts = zip(*valid_texts_with_indices)

    try:
        embeddings = embeddings_model.embed_documents(list(texts))

        for i, embedding in enumerate(embeddings):
            original_node_index = indices[i]
            nodes[original_node_index]["properties"]["embedding"] = embedding

    except Exception as e:
        print(f"ERRO: Falha ao gerar embeddings em lote: {e}")

    return graph_data

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Divide o texto em chunks menores usando um text splitter."""
    print(f"-> Dividindo o texto em chunks de até {chunk_size} caracteres...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

def generate_chunk_embeddings(chunks: list[str], embeddings_model: OpenAIEmbeddings) -> list[dict]:
    """Gera embeddings para uma lista de textos (chunks) e retorna uma lista de dicionários."""
    if not chunks:
        return []
    print(f"-> Gerando embeddings para {len(chunks)} chunks de texto...")
    try:
        embeddings = embeddings_model.embed_documents(chunks)
        return [{"text": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
    except Exception as e:
        print(f"ERRO: Falha ao gerar embeddings para os chunks: {e}")
        return [{"text": chunk, "embedding": None} for chunk in chunks]

def import_graph_to_neo4j(tx, scraping_id: int, document_title: str, chunks_with_embeddings: list, entities: list, relationships: list):
    """
    Função de transação para importar o grafo completo para o Neo4j,
    incluindo Documento, Chunks com links sequenciais, Entidades e todas as suas conexões.
    
    [VERSÃO CORRIGIDA DOS BUGS DE CYPHER]
    """
    tx.run(
        """
        MERGE (d:Documento {source_scraping_id: $scraping_id})
        SET d.titulo = $title, d.last_processed = timestamp()
        """,
        scraping_id=scraping_id, title=document_title
    )

    chunk_creation_query = """
    UNWIND $chunks AS chunk_data
    MATCH (d:Documento {source_scraping_id: $scraping_id})
    MERGE (c:Chunk {text: chunk_data.text})
    SET c.embedding = chunk_data.embedding
    MERGE (c)-[:PARTE_DE]->(d)
    """
    tx.run(chunk_creation_query, chunks=chunks_with_embeddings, scraping_id=scraping_id)
    
    if len(chunks_with_embeddings) > 1:
        sequential_link_query = """
        WITH $chunks AS chunks_list
        UNWIND range(0, size(chunks_list) - 2) AS i
        MATCH (c1:Chunk {text: chunks_list[i].text})
        MATCH (c2:Chunk {text: chunks_list[i+1].text})
        MERGE (c1)-[:PROXIMO_CHUNK]->(c2)
        """
        tx.run(sequential_link_query, chunks=chunks_with_embeddings)

    entity_creation_query = """
    UNWIND $entities AS entity_data
    MERGE (e:Entidade {nome: entity_data.properties.nome})
    
    WITH e, entity_data  /* <--- CORREÇÃO 2: Adicionado o "WITH" faltante */
    
    CALL apoc.create.addLabels(e, [entity_data.label]) YIELD node
    SET node += entity_data.properties
    RETURN entity_data.id AS temp_id, elementId(node) AS graph_element_id
    """
    result = tx.run(entity_creation_query, entities=entities)
    id_mapping = {record["temp_id"]: record["graph_element_id"] for record in result}

    if entities:
        mentions_creation_query = """
        UNWIND $entities AS entity_data
        MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS toLower(entity_data.properties.nome)
        MATCH (e:Entidade {nome: entity_data.properties.nome})
        MERGE (c)-[:MENCIONA]->(e)
        """
        tx.run(mentions_creation_query, entities=entities)

    if relationships and id_mapping:
        domain_rel_query = """
        UNWIND $rels AS rel
        MATCH (source) WHERE elementId(source) = $id_mapping[rel.source_id]
        MATCH (target) WHERE elementId(target) = $id_mapping[rel.target_id]
        CALL apoc.merge.relationship(source, rel.type, {}, rel.properties, target) YIELD rel AS created_rel
        RETURN count(created_rel)
        """
        tx.run(domain_rel_query, rels=relationships, id_mapping=id_mapping)

def fetch_unprocessed_news(pg_cursor, neo4j_driver):
    """Busca notícias do PostgreSQL que ainda não foram processadas no Neo4j."""
    print("-> Verificando notícias já processadas no Neo4j...")
    with neo4j_driver.session(database="neo4j") as session:
        result = session.run("MATCH (d:Documento) WHERE d.source_scraping_id IS NOT NULL RETURN d.source_scraping_id AS id")
        processed_ids = {record["id"] for record in result}

    print(f"   - {len(processed_ids)} documentos já existem no grafo.")

    print("-> Buscando TODAS as notícias novas para processar...")
    pg_cursor.execute("""
        SELECT id, titulo, texto 
        FROM scraping 
        WHERE texto IS NOT NULL AND LENGTH(texto) > 100
        ORDER BY id
    """)
    all_news = pg_cursor.fetchall()

    unprocessed_news = [news for news in all_news if news[0] not in processed_ids]
    print(f"   - {len(unprocessed_news)} novas notícias encontradas para processar.")
    return unprocessed_news

if __name__ == "__main__":
    print("--- Iniciando Processo de Ingestão de PostgreSQL para Grafo ---")
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("A variável de ambiente OPENAI_API_KEY não foi definida.")

    llm_client = OpenAI(api_key=openai_api_key)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    pg_conn = None
    neo4j_driver = None
    try:
        pg_conn = psycopg2.connect(
            dbname=os.getenv("SCRAP_PG_DATABASE"),
            user=os.getenv("SCRAP_PG_USER"),
            password=os.getenv("SCRAP_PG_PASSWORD"),
            host=os.getenv("SCRAP_PG_HOST"),
            port=os.getenv("SCRAP_PG_PORT")
        )
        pg_cursor = pg_conn.cursor()
        print("-> Conectado ao PostgreSQL (wscrap_db).")

        NEO4J_URI = os.getenv("NEO4J_URI")
        NEO4J_USER = os.getenv("NEO4J_USER")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
            raise ValueError("As variáveis de ambiente do Neo4j (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) não foram definidas.")

        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print(f"-> Conectado ao Neo4j em {NEO4J_URI}.")

        print("-> Garantindo a existência do índice vetorial 'chunk_embeddings'...")
        with neo4j_driver.session(database="neo4j") as session:
            session.run("""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                  indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                  }
                }
            """)
        print("   - Índice vetorial pronto.")

        news_to_process = fetch_unprocessed_news(pg_cursor, neo4j_driver)
        
        processed_count = 0
        pause_interval = 15
        pause_duration = 60

        for scraping_id, title, text in news_to_process:
            print(f"\n--- Processando Notícia ID: {scraping_id} | Título: {title[:50]}... ---")

            text_chunks = chunk_text(text)
            if not text_chunks:
                print("   - AVISO: Não foi possível dividir o texto em chunks. Pulando.")
                continue

            chunks_with_embeddings = generate_chunk_embeddings(text_chunks, embeddings_model)

            graph_data = extract_graph_from_text(text, llm_client)
            
            graph_data_with_node_embeddings = add_embeddings_to_nodes(graph_data, embeddings_model)

            if not graph_data_with_node_embeddings or not graph_data_with_node_embeddings.get("nodes"):
                print("   - AVISO: Nenhuma estrutura de grafo foi extraída. Continuando apenas com chunks.")
                entities = []
                relationships = [] 
            else:
                entities = graph_data_with_node_embeddings.get("nodes",)
                relationships = graph_data_with_node_embeddings.get("relationships",)

            with neo4j_driver.session(database="neo4j") as session:
                session.execute_write(
                    import_graph_to_neo4j,
                    scraping_id,
                    title,
                    chunks_with_embeddings,
                    entities,
                    relationships
                )
            print(f"   - Grafo da notícia ID {scraping_id} importado com sucesso!")
            
            processed_count += 1
            if processed_count % pause_interval == 0 and len(news_to_process) > processed_count:
                print(f"   - Processadas {processed_count} notícias. Pausando por {pause_duration} segundos para evitar rate limits...")
                time.sleep(pause_duration)


        print("\n--- Processo de ingestão para o grafo concluído! ---")

    except Exception as e:
        print(f"\nOcorreu um erro geral durante a ingestão: {e}")
    finally:
        if pg_conn:
            pg_conn.close()
            print("\n-> Conexão com PostgreSQL fechada.")
        if neo4j_driver:
            neo4j_driver.close()
            print("-> Conexão com Neo4j fechada.")