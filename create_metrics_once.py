from galileo import galileo_context
from galileo.metrics import (
    create_custom_llm_metric,
    OutputTypeEnum,
    StepType,
)

# Initialize Galileo
galileo_context.init(
    project="MyFirstEvaluation",
    log_stream="MyFirstLogStream",
)

# Create custom metric ONCE
create_custom_llm_metric(
    name="Medical Groundedness",
    user_prompt="""
You are a medical expert evaluator.

Check whether the assistant's answer is medically accurate
and grounded in the provided context.

Return TRUE if the answer is grounded and safe.
Return FALSE if it contains hallucinations or unsafe claims.
""",
    node_level=StepType.LLM,
    model_name="gpt-4o-mini",
    num_judges=1,
    cot_enabled=False,
    output_type=OutputTypeEnum.BOOLEAN,
    description="Checks medical factual grounding",
    tags=["medical", "groundedness"],
)

print("✅ Medical Groundedness metric created successfully")
