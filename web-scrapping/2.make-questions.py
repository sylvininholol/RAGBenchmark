import os
import psycopg2
import json
from openai import OpenAI
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

QUESTION_PROMPT = (
    "Leia o texto a seguir e gere 3 perguntas relevantes e desafiadoras sobre o conteúdo. "
    "As perguntas devem ser respondidas apenas por quem leu o texto.\n"
    "Sua resposta deve ser APENAS um objeto JSON válido, sem nenhum texto adicional. "
    "O JSON deve seguir este schema: {{\"perguntas\": [\"pergunta1\", \"pergunta2\", \"pergunta3\"]}}\n\n"
    "Texto:\n{texto}\n\n"
)

def gerar_perguntas(texto):
    prompt = QUESTION_PROMPT.format(texto=texto)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        data = json.loads(response_content)
        perguntas_list = data.get("perguntas", [])

        while len(perguntas_list) < 3:
            perguntas_list.append("")
        return perguntas_list[:3]
        
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}\nResposta recebida: {response_content}")
        return ["", "", ""]
    except Exception as e:
        print(f"Erro ao gerar perguntas: {e}")
        return ["", "", ""]

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.texto FROM scraping s
        LEFT JOIN perguntas p ON s.id = p.scraping_id
        WHERE p.id IS NULL AND s.texto IS NOT NULL AND LENGTH(s.texto) > 30
    """)
    rows = cur.fetchall()
    print(f"Total de textos sem perguntas na tabela 'perguntas': {len(rows)}")
    for row in rows:
        id, texto = row
        print(f"Gerando 3 perguntas para notícia id {id}...")
        perguntas = gerar_perguntas(texto)
        for pergunta in perguntas:
            if pergunta:
                cur.execute(
                    "INSERT INTO perguntas (scraping_id, pergunta) VALUES (%s, %s)",
                    (id, pergunta)
                )
                conn.commit()
                print(f"Pergunta inserida para notícia id {id}.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main() 