from rag_cli import run_rag
from evaluations.test_cases import TEST_CASES

print("Running Galileo Evaluations...\n")

for i, case in enumerate(TEST_CASES, start=1):
    print(f"Test {i}: {case['question']}")
    answer = run_rag(case["question"])
    print("Answer:", answer)
    print("-" * 50)
