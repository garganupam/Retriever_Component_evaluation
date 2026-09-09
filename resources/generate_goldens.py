import os
import re
import glob
import json
import random

from dotenv import load_dotenv

from deepeval.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_groq import ChatGroq
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


load_dotenv()


# --------------------------------------------------
# 1. GROQ MODEL
# --------------------------------------------------

class GroqModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatGroq(
            model="groq/compound-mini",
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
        return "groq/compound-mini"


groq_model = GroqModel()


# --------------------------------------------------
# 2. LOAD + CHUNK YOUR VTT FILES
# --------------------------------------------------

def load_chunks():

    texts = []

    for path in glob.glob("data/*.vtt"):

        with open(path) as f:

            lines = [
                ln.strip()
                for ln in f
                if ln.strip()
                and ln.strip() != "WEBVTT"
                and "-->" not in ln
            ]

        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text(
        "\n\n".join(texts)
    )


# --------------------------------------------------
# 3. SAMPLE CHUNKS
# --------------------------------------------------

chunks = load_chunks()

sample = random.sample(
    chunks,
    min(1, len(chunks))
)

contexts = [
    [chunk]
    for chunk in sample
]


# --------------------------------------------------
# 4. GENERATE GOLDENS USING GROQ
# --------------------------------------------------

synthesizer = Synthesizer(
    model=groq_model
)

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)


# --------------------------------------------------
# 5. CONVERT TO YOUR JSON FORMAT
# --------------------------------------------------

rows = []

for i, g in enumerate(goldens, 1):

    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",
    })


# --------------------------------------------------
# 6. SAVE
# --------------------------------------------------

os.makedirs("goldens", exist_ok=True)

with open(
    "goldens/retriever_deepeval_goldens.json",
    "w"
) as f:

    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"wrote {len(rows)} DRAFT goldens "
    f"-> goldens/retriever_deepeval_goldens.json"
)

print(
    "!! REVIEW EVERY ONE before using: "
    "check grounding, trim padding, fix leading questions."
)