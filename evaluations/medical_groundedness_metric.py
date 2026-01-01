from galileo.metrics import (
    create_custom_llm_metric,
    OutputTypeEnum,
    StepType
)

medical_groundedness_metric = create_custom_llm_metric(
    name="Medical Answer Groundedness",
    user_prompt="""
You are an impartial medical AI evaluator.

Task:
Determine whether the LLM answer strictly uses the provided CONTEXT
and does NOT introduce new medical facts, symptoms, diagnoses, or
recommendations.

Definitions:
- Grounded: All claims are explicitly supported by the context.
- Not Grounded: Any symptom, explanation, or advice is added that
  is not found in the context.

Return true if the answer is fully grounded.
Return false otherwise.
""",
    node_level=StepType.llm,
    cot_enabled=True,
    model_name="gpt-4.1-mini",
    num_judges=3,
    output_type=OutputTypeEnum.BOOLEAN,
    description="Checks whether healthcare answers are strictly grounded in retrieved context",
    tags=["healthcare", "rag", "hallucination"]
)
