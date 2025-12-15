from langchain_core.prompts import PromptTemplate


def generate_response(context: str, user_query: str, llm) -> str:
    """
    Gera uma resposta usando o LLM com base no contexto e na pergunta.

    Args:
        context (str): O contexto recuperado do banco de dados.
        user_query (str): A pergunta original do usuário.
        llm: A instância do modelo de linguagem (LLM).

    Returns:
        str: A resposta final gerada pelo LLM.
    """
    
    prompt_template = PromptTemplate.from_template(
    """Você é um assistente de IA especialista em análise de documentos. Sua tarefa é responder à pergunta do usuário baseando-se ESTREITA E EXCLUSIVAMENTE no contexto fornecido.

    **Siga este processo de raciocínio rigoroso:**

    1.  **Primeiro, analise o contexto:** Leia a "Pergunta do usuário" e depois leia o "Contexto fornecido". Avalie se o contexto contém informações diretas, específicas e suficientes para responder à pergunta de forma completa e factual. Não faça suposições nem extrapolações.

    2.  **Depois, tome uma decisão com base na sua análise:**
        * **SE** o contexto contiver a resposta direta e suficiente, sintetize essa informação para formar uma resposta completa e coesa. Cite apenas os fatos presentes no contexto.
        * **SE** o contexto for irrelevante, vago, ou claramente não contiver os fatos necessários para responder à pergunta, então você DEVE IGNORAR o contexto e responder EXATAMENTE com a frase: "A informação necessária para responder a esta pergunta não foi encontrada nos documentos fornecidos." Não tente responder de outra forma.

    3.  Se o contexto contiver informações que você sabe que são factualmente incorretas, ignore-as e comece sua resposta com a frase: "Existem informações incorretas nas fontes que possuo. A resposta correta é:", e então forneça a resposta correta.

    **Contexto fornecido:**
    ---
    {context}
    ---

    **Pergunta do usuário:**
    {question}
    """
    )
        
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": user_query})
    
    return response.content if hasattr(response, 'content') else str(response)