# 🚀 Benchmark de Arquiteturas RAG para Notícias Esportivas

Este repositório contém a implementação e o *benchmark* de quatro arquiteturas de Geração Aumentada por Recuperação (**RAG**: *Retrieval Augmented Generation*) para um estudo de caso focado em informações factuais e dinâmicas (notícias esportivas). O projeto utiliza containers Docker (PostgreSQL/pgvector e Neo4j) e a API da OpenAI para execução e avaliação.

## 🗃️ Arquitetura do Projeto

O projeto está dividido em quatro pipelines de arquitetura e um módulo de preparação de dados (Scrapping).

| Diretório | Arquitetura | Base de Conhecimento (RK) | Descrição do Pipeline |
| :--- | :--- | :--- | :--- |
| `baseline-llm` | **Baseline (LLM Puro)** | PostgreSQL (Apenas Resultados) | O LLM (GPT-3.5-Turbo) responde sem recuperação de contexto, simulando a limitação de conhecimento estático. |
| `naive-rag` | **Naive RAG** | PostgreSQL (PGVector) | Recuperação simples de `chunks` de texto por similaridade vetorial (`top_k`). |
| `advanced-rag` | **Advanced RAG** | PostgreSQL (PGVector) | Combina **busca por resumos** de documentos e **re-ranking** dos *chunks* recuperados (Sumarização + Re-ranking). |
| `graph-rag` | **Graph RAG** | Neo4j (Grafo) e PGSQL | Recuperação híbrida (vetor + entidades) e expansão de contexto via grafo, traduzido para linguagem natural via LLM. |
| `web-scrapping` | **Dataset & QA Generation**| PostgreSQL (Scrapping DB) | Módulo de coleta de notícias e geração do conjunto de dados de avaliação (P&R Simples, Multi-Contexto e Rejeição Negativa). |

---

## ⚙️ Configuração do Ambiente

### 1. Pré-requisitos

* **Docker** e **Docker Compose** (Necessário para todos os serviços de banco de dados).
* **Python** (Versão 3.9+).
* **Chave da API da OpenAI** (`sk-proj-XXXX...`).

### 2. Instalação das Dependências Python

Instale os pacotes listados em `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Configuração do Arquivo .env

Crie um arquivo chamado **`.env`** na **raiz** do projeto. Ele deve conter as credenciais de conexão para todos os serviços, espelhando as configurações definidas nos arquivos `docker-compose.yml`.

**IMPORTANTE:** Substitua os valores de `POSTGRES_PASSWORD` e `NEO4J_PASSWORD` pelos valores reais que você usará nos seus arquivos `docker-compose.yml`.

| Variável | Valor Exemplo (Baseado no `docker-compose`) | Descrição |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | `sk-proj-SEUVALORAQUI...` | Chave da API para Embeddings e LLMs |
| **--- Naive RAG (Porta 5432) ---** | | |
| `PG_HOST` | `127.0.0.1` | Host |
| `PG_DATABASE` | `tcc_db` | Nome do DB |
| `PG_USER` | `bancoRAG` | Usuário |
| `PG_PASSWORD` | `senha123` | **Senha (MUDAR!)** |
| `PG_PORT` | `5432` | Porta de acesso |
| **--- Scrapping DB (Porta 5433) ---** | | |
| `SCRAP_PG_DATABASE` | `wscrap_db` | Nome do DB |
| `SCRAP_PG_USER` | `wscrap_user` | Usuário |
| `SCRAP_PG_PASSWORD` | `wscrap_pass` | **Senha (MUDAR!)** |
| `SCRAP_PG_HOST` | `127.0.0.1` | Host |
| `SCRAP_PG_PORT` | `5433` | Porta de acesso |
| **--- Baseline DB (Porta 5430) ---** | | |
| `BASELINE_PG_DATABASE` | `baseline_db` | Nome do DB |
| `BASELINE_PG_USER` | `bancoBaseline` | Usuário |
| `BASELINE_PG_PASSWORD` | `senha123` | **Senha (MUDAR!)** |
| `BASELINE_PG_HOST` | `127.0.0.1` | Host |
| `BASELINE_PG_PORT` | `5430` | Porta de acesso |
| **--- Advanced RAG DB (Porta 5434) ---** | | |
| `ADV_PG_DATABASE` | `adv_rag_db` | Nome do DB |
| `ADV_PG_USER` | `adv_rag_user` | Usuário |
| `ADV_PG_PASSWORD` | `adv_rag_password` | **Senha (MUDAR!)** |
| `ADV_PG_HOST` | `127.0.0.1` | Host |
| `ADV_PG_PORT` | `5434` | Porta de acesso |
| **--- Neo4j (Graph RAG) ---** | | |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | URI de conexão |
| `NEO4J_USER` | `neo4j` | Usuário |
| `NEO4J_PASSWORD` | `1zc-WQh61g9abEjbDY9WatMXsAsm32HckKL1ikJQf0k` | **Senha (MUDAR!)** |
| **--- Graph RAG Evaluation DB (Porta 5429) ---** | | |
| `GRAPH_PG_USER` | `graph_user` | Usuário |
| `GRAPH_PG_PASSWORD` | `graph_pass` | **Senha (MUDAR!)** |
| `GRAPH_PG_DATABASE` | `graph_rag_db` | Nome do DB |
| `GRAPH_PG_HOST` | `localhost` | Host |
| `GRAPH_PG_PORT` | `5429` | Porta de acesso |

### 4. Inicialização dos Bancos de Dados com Docker

Todos os bancos de dados PostgreSQL (`pgvector`) e o Neo4j devem ser iniciados antes de qualquer script Python ser executado.

#### Iniciar todos os serviços (Naive, Baseline, Advanced e Graph):

Use os comandos abaixo para iniciar os containers de cada arquitetura, conforme configurado nos respectivos `docker-compose.yml`:

```bash
# Inicia Naive RAG (PostgreSQL na Porta 5432)
cd naive-rag && sudo docker compose up -d

# Inicia Baseline (PostgreSQL na Porta 5430)
cd ../baseline-llm && sudo docker compose up -d

# Inicia Advanced RAG (PostgreSQL na Porta 5434)
cd ../advanced-rag && sudo docker compose up -d

# Inicia Graph RAG (Neo4j: 7687, PostgreSQL de Avaliação: 5429)
cd ../graph-rag && sudo docker compose up -d
```

#### Configurar Índices Vetoriais no Neo4j:

Após iniciar o Neo4j (acessível em `http://localhost:7474`), execute o seguinte Cypher no Neo4j Browser. Este índice é crucial para a recuperação vetorial dos chunks na arquitetura Graph RAG:

```Cypher
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

## 🏃 Fluxo de Execução (Reprodução do Benchmark)

### Fase 1: Criação do Dataset de Avaliação (`web-scrapping`)

O `wscrap_db` (porta 5433) é o *schema* central de onde todos os dados de avaliação serão puxados.

1.  **Coleta de Notícias (Scraping):** Popula a tabela `scraping`.

    ```bash
    cd web-scrapping
    python 1.scrapper_ge.py
    ```

2.  **Geração de Perguntas Simples:** Cria perguntas na tabela `perguntas`.

    ```bash
    python 2.make-questions.py
    ```

3.  **Geração de Embeddings e Respostas Padrão (Ground Truth):** Gera respostas e *embeddings* de referência para as perguntas simples.

    ```bash
    python 3.generate-embeddings-and-responses.py
    ```

4.  **Geração de P&R Complexas (Multi-Contexto e Rejeição Negativa):** Adiciona as perguntas mais complexas de avaliação, forçando a síntese de informação ou a recusa de resposta.

    ```bash
    python 4.generate_advanced_qa.py
    ```

### Fase 2: Ingestão de Conhecimento (RK)

Os dados do `wscrap_db` são migrados e transformados nas bases de conhecimento de cada arquitetura.

1.  **Ingestão Naive RAG:** Divide o texto em *chunks* e armazena com *embeddings*.

    ```bash
    cd ../naive-rag
    python 1.generate_knowledge_base.py
    ```

2.  **Ingestão Advanced RAG:** Além dos *chunks*, gera e armazena resumos de documentos com *embeddings*.

    ```bash
    cd ../advanced-rag
    python 1.generate_knowledge_base.py
    ```

3.  **Ingestão Graph RAG:** Transforma o texto em *chunks* e, usando LLM, em entidades e relacionamentos no Neo4j.

    ```bash
    cd ../graph-rag
    python 1.generate_knowledge_base.py
    ```

### Fase 3: Execução e Avaliação do Pipeline

Execute o script `2.evaluate_rag.py` em cada diretório. Ele buscará as perguntas, executará o pipeline RAG/Baseline, calculará as métricas (Similaridade, LLM Judge, RAGAS) e salvará os resultados na tabela `evaluation_results` do respectivo banco de dados.

| Arquitetura | Comando de Avaliação |
| :--- | :--- |
| **Baseline (LLM Puro)** | `cd ../baseline-llm && python 2.evaluate_rag.py` |
| **Naive RAG** | `cd ../naive-rag && python 2.evaluate_rag.py` |
| **Advanced RAG** | `cd ../advanced-rag && python 2.evaluate_rag.py` |
| **Graph RAG** | `cd ../graph-rag && python 2.evaluate_rag.py` |

### Fase 4: Teste de Inferência e Análise

#### Testes de Inferência Simples (Opcional):

Execute para testar se a recuperação e a geração estão funcionando corretamente em cada arquitetura:

```bash
# Testar Naive RAG
python naive-rag/inference.py

# Testar Advanced RAG
python advanced-rag/inference.py

# Testar Graph RAG
python graph-rag/inference.py
```

#### Análise do Dataset (Opcional):

Gera gráficos e estatísticas sobre a distribuição e o conteúdo do conjunto de dados.

```bash
cd ../web-scrapping
python 5.analyze_dataset.py
```