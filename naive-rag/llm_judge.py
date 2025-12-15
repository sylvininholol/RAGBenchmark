import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

JUDGE_LLM = None

def get_judge_llm(api_key: str):
    """Inicializa o LLM Juiz de forma preguiçosa (lazy)."""
    global JUDGE_LLM
    if JUDGE_LLM is None:
        JUDGE_LLM = ChatOpenAI(model="gpt-5-nano", temperature=1, openai_api_key=api_key, model_kwargs={"response_format": {"type": "json_object"}})
    return JUDGE_LLM

EVALUATION_PROMPT_TEMPLATE = """
Você é um avaliador imparcial e rigoroso de sistemas de Resposta a Perguntas (Question Answering). Sua tarefa é avaliar a qualidade de uma resposta gerada por um sistema de IA, comparando-a com uma resposta "padrão ouro" (ground truth) que é considerada a verdade absoluta.

**Contexto da Avaliação:**
- Pergunta: {question}
- Resposta Padrão Ouro (Ground Truth): {ground_truth}
- Resposta Gerada para Avaliar: {generated_answer}

**Critérios de Avaliação:**
1.  **Correção e Fidelidade:** A resposta gerada está factualmente alinhada com a resposta padrão ouro? Ela evita alucinações ou informações contraditórias?
2.  **Completude:** A resposta gerada aborda todos os aspectos importantes contidos na resposta padrão ouro?

**Instruções de Pontuação (1 a 5):**
- 5: Excelente. A resposta gerada é factualmente correta, completa e alinhada com a resposta padrão ouro.
- 4: Boa. A resposta é correta, mas pode ter pequenas omissões ou informações extras que não prejudicam a correção geral.
- 3: Razoável. A resposta está parcialmente correta, mas omite detalhes importantes ou contém imprecisões menores.
- 2: Ruim. A resposta contém erros factuais significativos ou é muito incompleta.
- 1: Péssima. A resposta é completamente incorreta, irrelevante ou alucina informações.

**Formato da Saída:**
Sua saída DEVE ser um único objeto JSON, e nada mais. O formato deve ser:
{{"score": <sua_pontuação_de_1_a_5>, "reasoning": "<sua_justificativa_detalhada_e_concisa>"}}

**JSON de Avaliação:**
"""
EVALUATION_PROMPT = PromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)

def evaluate_with_llm_judge(question: str, ground_truth_answer: str, generated_answer: str, api_key: str) -> tuple[int | None, str | None]:
    """
    Usa um LLM poderoso para julgar a qualidade da resposta gerada.
    Retorna uma tupla (score, reasoning).
    """
    try:
        judge = get_judge_llm(api_key)
        chain = EVALUATION_PROMPT | judge

        response = chain.invoke({
            "question": question,
            "ground_truth": ground_truth_answer,
            "generated_answer": generated_answer
        })

        result_json = json.loads(response.content)

        score = result_json.get("score")
        reasoning = result_json.get("reasoning")

        return score, reasoning

    except json.JSONDecodeError as e:
        print(f"   - ERRO (LLM Judge): Falha ao decodificar a resposta JSON. {e}")
        return None, f"Erro de decodificação JSON: {response.content}"
    except Exception as e:
        print(f"   - ERRO (LLM Judge): Falha na chamada da API. {e}")
        return None, f"Erro na API: {e}"
