CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    question_id INTEGER NOT NULL UNIQUE,
    question_text TEXT,
    ground_truth_answer_embedding TEXT,
    rag_answer_text TEXT,
    rag_answer_embedding TEXT,
    retrieved_context TEXT,
    cosine_similarity FLOAT,
    model_name VARCHAR(100),
    llm_judge_score INTEGER,
    llm_judge_reasoning TEXT,
    ragas_faithfulness FLOAT,
    ragas_context_relevancy FLOAT, -- Renomeado de context_precision
    ragas_context_recall FLOAT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_question_id ON evaluation_results (question_id);