import os
import json
import time

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_groq import ChatGroq

from src.retriever import build_retriever


load_dotenv()


# ============================================================
# CONFIG
# ============================================================

GOLDEN_PATH = "goldens/retriever_goldens.json"

# Groq model used as the DeepEval judge
JUDGE_MODEL = "groq/compound-mini"

# Minimum score required to pass a metric
THRESHOLD = 0.7

# Local Hugging Face embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of questions evaluated in one batch
BATCH_SIZE = 1

# Wait time between batches
WAIT_SECONDS = 10


# ============================================================
# GROQ MODEL WRAPPER FOR DEEPEVAL
# ============================================================

class GroqModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatGroq(
            model=JUDGE_MODEL,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return JUDGE_MODEL


# Create Groq judge
groq_judge = GroqModel()


# ============================================================
# LOAD GOLDENS
# ============================================================

def load_goldens(path):

    with open(path, encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CREATE TEST CASES
# ============================================================

def create_test_cases(goldens, retriever):

    test_cases = []

    for g in goldens:

        print(f"Retrieving context for: {g['query']}")

        retrieved = retriever.invoke(g["query"])

        retrieval_context = [
            doc.page_content
            for doc in retrieved
        ]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,

                # Generator is not being evaluated.
                actual_output="(generator not evaluated in this run)",
            )
        )

    return test_cases


# ============================================================
# EVALUATE ONE BATCH
# ============================================================

def evaluate_batch(test_cases, batch_number, total_batches):

    print()
    print("=" * 70)
    print(f"Evaluating batch {batch_number}/{total_batches}")
    print(f"Questions in this batch: {len(test_cases)}")
    print("=" * 70)
    print()

    # Create fresh metric objects for each batch
    metrics = [

        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=groq_judge,
            include_reason=False,
        ),

        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=groq_judge,
            include_reason=False,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,

        hyperparameters={
            "retriever": "base_k5",

            "embedding_model": EMBEDDING_MODEL,

            "chunk_size": 1000,
            "chunk_overlap": 200,

            "top_k": 5,

            "judge_model": JUDGE_MODEL,

            "golden_set": GOLDEN_PATH,

            "batch_size": BATCH_SIZE,

            "wait_between_batches": WAIT_SECONDS,
        },
    )

    return result


# ============================================================
# RUN COMPLETE EVALUATION
# ============================================================

def run(retriever):

    # --------------------------------------------------------
    # 1. LOAD ALL GOLDENS
    # --------------------------------------------------------

    goldens = load_goldens(GOLDEN_PATH)

    total_questions = len(goldens)

    print()
    print("=" * 70)
    print(f"Total golden questions: {total_questions}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Wait between batches: {WAIT_SECONDS} seconds")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 2. CREATE ALL TEST CASES
    # --------------------------------------------------------

    test_cases = create_test_cases(
        goldens,
        retriever,
    )

    # --------------------------------------------------------
    # 3. SPLIT INTO BATCHES
    # --------------------------------------------------------

    batches = [
        test_cases[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(test_cases),
            BATCH_SIZE,
        )
    ]

    total_batches = len(batches)

    print()
    print(f"Created {total_batches} batches.")
    print()

    # --------------------------------------------------------
    # 4. EVALUATE EACH BATCH
    # --------------------------------------------------------

    results = []

    for i, batch in enumerate(batches):

        batch_number = i + 1

        result = evaluate_batch(
            batch,
            batch_number,
            total_batches,
        )

        results.append(result)

        # ----------------------------------------------------
        # WAIT BEFORE NEXT BATCH
        # ----------------------------------------------------

        if batch_number < total_batches:

            print()
            print("-" * 70)
            print(
                f"Batch {batch_number} finished."
            )
            print(
                f"Waiting {WAIT_SECONDS} seconds "
                "before the next batch..."
            )
            print("-" * 70)
            print()

            time.sleep(WAIT_SECONDS)

    # --------------------------------------------------------
    # 5. COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL BATCHES COMPLETED")
    print("=" * 70)
    print()

    print(
        f"Evaluated {total_questions} "
        "golden questions successfully."
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Build retriever
    retriever = build_retriever()

    # Run complete evaluation
    results = run(retriever)

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()

    for i, result in enumerate(results, start=1):

        print(f"\nBatch {i} result:")
        print(result)