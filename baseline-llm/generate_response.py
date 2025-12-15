from langchain_core.prompts import PromptTemplate

def generate_response(user_query: str, llm) -> str:
    """
    Gera uma resposta usando o LLM diretamente, sem contexto de RAG.

    Args:
        user_query (str): A pergunta original do usuÃ¡rio.
        llm: A instÃ¢ncia do modelo de linguagem (LLM).

    Returns:
        str: A resposta gerada pelo LLM.
    """
    
    prompt_template = PromptTemplate.from_template(
    """Você é um assistente de IA. Sua tarefa é responder as perguntas do usuário da melhor forma possível.

    **Siga estas regras:**

    1.  Se você sabe a resposta, responda de forma clara e concisa.
    2.  **SE** você não tem certeza sobre a resposta ou se a pergunta é sobre informações muito específicas, proprietárias ou fora do seu conhecimento geral, você DEVE responder EXATAMENTE com a frase: "Não tenho informações para responder a esta pergunta." Não tente inventar uma resposta.

    **Pergunta do usuário:**
    {question}
    """
    )
        
    chain = prompt_template | llm
    response = chain.invoke({"question": user_query})
    
    return response.content if hasattr(response, 'content') else str(response)