# RAG

Implementação naive de RAG para o TCC. Deepseek API usada para responder aos prompts e OpenAI API usada para gerar embeddings. 



## Naive-rag

Para executar o naive rag pela primeira vez, no diretório do naive-rag faça:

```bash
sudo docker compose up
```

Após isso:

```bash
python db/db.py
```

Agora, pode rodar o script de inferencia para testar o llm com acesso ao banco, mas sem o banco populado:

```bash
python inference.py
```

A resposta deve ser:


```bash
A informação necessária para responder a esta pergunta não foi encontrada nos documentos fornecidos.
```

Para popular com o exemplo basico do curriculo:


```bash
python generate_knowledge_base.py
```

Agora ao repetir o script de inferencia deve receber:

```bash
Com base no contexto fornecido, a experiência profissional de Gabriel Marinho mencionada é:

**Software Engineer Intern na IBM Research | Dezembro de 2024 - Presente**
*   Membro da equipe de inferência do Watsonx, desenvolvendo soluções para inferência de modelos de linguagem grande (LLM) e contribuindo para o projeto de código aberto vLLM.
*   Possui forte conhecimento da arquitetura Transformers.
*   Tem experiência na construção de software modular e nas melhores práticas de design de sistemas.
```







# Iniciando Neo4j com Docker

Este guia rápido mostra como iniciar uma instância do Neo4j usando Docker e Docker Compose.

## Pré-requisitos

* Docker: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
* Docker Compose: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

## Passos

**1. Crie o arquivo `docker-compose.yml`:**

Crie um arquivo chamado `docker-compose.yml` no diretório do seu projeto com o seguinte conteúdo:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5 # Ou a versão mais recente/específica, ex: neo4j:5.18.1
    container_name: meu-neo4j
    ports:
      - "7474:7474" # Neo4j Browser (HTTP)
      - "7687:7687" # Bolt (protocolo do driver)
    volumes:
      - ./neo4j/data:/data
      - ./neo4j/logs:/logs
      - ./neo4j/import:/var/lib/neo4j/import
      - ./neo4j/plugins:/plugins # Para plugins como APOC
    environment:
      NEO4J_AUTH: neo4j/suaSenhaSuperSegura # Mude "suaSenhaSuperSegura"!
      # Para instalar o plugin APOC automaticamente (opcional):
      # NEO4J_PLUGINS: '["apoc"]'
    restart: unless-stopped
```
**Importante:** Altere `suaSenhaSuperSegura` para uma senha forte.

**2. Crie os Diretórios Locais (APENAS CASO NAO SEJAM CRIADAS QUANDO RODAR O Neo4j):**

No mesmo diretório do seu `docker-compose.yml`, crie as pastas que serão usadas para os volumes, se ainda não existirem:
```bash
mkdir -p ./neo4j/data ./neo4j/logs ./neo4j/import ./neo4j/plugins
```
*Se estiver no Linux e for usar `sudo` para Docker, ajuste as permissões da pasta `./neo4j` se necessário:*
```bash
# sudo chown -R <span class="math-inline">\(id \-u\)\:</span>(id -g) ./neo4j
```

**3. Inicie o Neo4j:**

No terminal, no diretório do seu `docker-compose.yml` em graph-rag/, execute:
```bash
docker-compose up -d
```
*(Use `sudo docker-compose up -d` se o seu usuário não pertencer ao grupo `docker`)*

**4. Acesse o Neo4j Browser:**

Após alguns segundos, abra seu navegador e acesse:
[http://localhost:7474](http://localhost:7474)

Use as seguintes credenciais para conectar (conforme definido no `docker-compose.yml`):
* **Usuário:** `neo4j`
* **Senha:** `suaSenhaSuperSegura` (ou a que você definiu)

## Comandos Úteis do Docker Compose

(Execute no diretório do `docker-compose.yml`)

* **Parar o Neo4j:**
    ```bash
    docker-compose stop neo4j
    ```
* **Iniciar o Neo4j (após parado):**
    ```bash
    docker-compose start neo4j
    ```
* **Ver logs:**
    ```bash
    docker-compose logs -f neo4j
    ```
* **Parar e remover contêineres (dados nos volumes são preservados):**
    ```bash
    docker-compose down
    ```
* **Parar, remover contêineres E VOLUMES (CUIDADO: apaga os dados do Neo4j se os volumes não forem nomeados externamente):**
    ```bash
    # docker-compose down -v
    ```










# Salvar banco do NEO4J usando um dump

Passos para fazer backup (dump) do banco de dados local para compartilhamento via Git, e como restaurar/atualizar uma instância local a partir de um dump.

## Pré-requisitos

* Docker instalado: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
* Docker Compose instalado: [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

## Estrutura de Diretórios Esperada

```
graph-rag/
├── neo4j/
│   ├── data/         # (NÃO ADICIONAR AO GIT)
│   ├── import/       # (ADICIONAR DUMPS AO GIT AQUI)
│   ├── logs/         # (NÃO ADICIONAR AO GIT)
│   └── plugins/
└── docker-compose.yml
```

## Parte 1: Fazendo Backup (Dump) do Banco Local para o Git

**Passo 1: Preparar Permissões de Diretório (SEMPRE QUE TIVER ERROS DE PERMISSÃO USE)**

No terminal, na raiz do projeto:
```bash
sudo chown -R $(id -u):$(id -g) ./neo4j
```
Isso é necessário devido a um problema de permissões que irá ser resolvido no futuro. 

**Passo 2: Parar Neo4j**
```bash
sudo docker-compose stop neo4j
```

**Passo 3: Criar Dump**

O nome do arquivo incluirá data/hora.
```bash
sudo docker-compose run --rm neo4j neo4j-admin database dump neo4j --to-stdout > ./neo4j/import/neo4j_backup_$(date +%Y%m%d_%H%M%S).dump
```

**Passo 4: Verificar Dump**
```bash
ls -lh ./neo4j/import/
```

**Passo 5: Adicionar ao Git e Enviar**
Substitua `neo4j_backup_YYYYMMDD_HHMMSS.dump` pelo nome real do arquivo.
```bash
git add docker-compose.yml .gitignore ./neo4j/import/neo4j_backup_YYYYMMDD_HHMMSS.dump
git commit -m "Adiciona backup Neo4j $(date +%d-%m-%Y)"
git push
```
Ou apenas adicione pelo vscode. Caso não apareça em neo4j/imports, repita o sudo chown do inicio.

**Passo 6: Reiniciar Neo4j (Local)**
```bash
sudo docker-compose start neo4j
# ou: sudo docker-compose up -d
```

## Parte 2: Restaurando/Atualizando Banco Local a partir de um Dump do Git

**Passo 1: Obter Atualizações do Repositório**
```bash
git pull
```
O arquivo de dump deverá estar em `./neo4j/import/`.

**Passo 2: Parar Neo4j**

Se uma instância Neo4j já estiver rodando:
```bash
sudo docker-compose stop neo4j
```

**Passo 3: Carregar Dados do Dump**

Substitua `NOME_DO_ARQUIVO_DE_DUMP.dump` pelo nome real do arquivo.
```bash
sudo docker-compose run --rm neo4j neo4j-admin database load neo4j --from-path=/var/lib/neo4j/import/NOME_DO_ARQUIVO_DE_DUMP.dump --overwrite-destination=true
```

**Passo 4: Iniciar Neo4j com Dados Carregados**
```bash
sudo docker-compose up -d
```
Acesse em `http://localhost:7474`.

## Observações sobre Permissões e `sudo`

* Comandos usam `sudo`. Para evitar no Linux: `sudo usermod -aG docker $USER` (requer logout/login ou reboot).
* Se problemas de permissão em `./neo4j/import` (ex: arquivos não visíveis no Git/VSCode):
    ```bash
    sudo chown -R $(id -u):$(id -g) ./neo4j
    ```
  Execute antes de criar dumps ou após `git pull` se necessário.


## Com o Banco em Pé, rode no neo4j browser a seguinte linha de comando

CREATE VECTOR INDEX `global-embedding-index` IF NOT EXISTS
FOR (n:Searchable) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}

CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}

Ela serve para criar um índice global

## Para verificar se funcionou rode:

## Caso precise subir o docker novamente com mudanças em plugins, etc

sudo rm -rf ./neo4j/data/*