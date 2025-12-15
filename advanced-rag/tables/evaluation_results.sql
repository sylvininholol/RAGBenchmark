CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    question_id INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    ground_truth_answer_embedding VECTOR(1536),
    rag_answer_text TEXT,
    rag_answer_embedding VECTOR(1536),
    retrieved_context TEXT,
    cosine_similarity DOUBLE PRECISION,
    model_name VARCHAR(255),
    llm_judge_score INTEGER,
    llm_judge_reasoning TEXT,
    ragas_faithfulness DOUBLE PRECISION,
    ragas_context_relevancy DOUBLE PRECISION,
    ragas_context_recall DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_evaluation_question_id ON evaluation_results(question_id);