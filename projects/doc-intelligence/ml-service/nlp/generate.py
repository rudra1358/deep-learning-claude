"""Generative answers with citations (Retrieval-Augmented Generation, RAG).

The extractive model can only copy a short span. For "summarise the termination rules" a
written answer is better, so a small instruction-tuned LLM (Qwen2.5-1.5B-Instruct, runs on a
6 GB GPU in fp16) writes the answer - but ONLY from the retrieved chunks:

  1. the top chunks are numbered [1], [2], ... in the prompt
  2. the system prompt says: answer only from these sources, cite them like [2], and say
     "I could not find this in the documents" otherwise
  3. we parse the [n] markers back into links to the real chunks (page + document)

Grounding + citations is what separates a document assistant from a generic chatbot: the user
can click every claim and check it. The model can still make mistakes, which is why the UI
always shows the sources next to the answer and why the extractive mode exists.
"""
from __future__ import annotations

import re

import torch

from .common import pick_device

NOT_FOUND = "I could not find this in the documents."

# v1: instructions only. Measured: the 1.5 B model cited a source in only 24 % of answers and invented an
#     answer for 87 % of unanswerable questions - small models follow abstract rules poorly.
SYSTEM_V1 = ("You answer questions using ONLY the numbered sources provided. Cite every sentence with the "
             "source numbers in square brackets, e.g. [1] or [2][3]. If the sources do not contain the answer, "
             f"reply exactly: {NOT_FOUND}")
# v2: explicit rules incl. the false-premise trap (question has a detail the sources contradict), plus two
#     worked examples ("few-shot") - small models copy demonstrated behaviour much better than instructions.
SYSTEM_V2 = ("You are a careful assistant. Answer ONLY with facts stated in the numbered sources.\n"
             "Rules:\n1. After every sentence, write the numbers of the sources you used, like [2].\n"
             "2. If the sources do not state the answer, or the question contains a name, date or detail that "
             f"does not match the sources, reply exactly: {NOT_FOUND}\n3. Never guess and never correct the question.")
FEW_SHOT = [
    {"role": "user", "content": "Sources:\n[1] The Eiffel Tower was completed in 1889.\n[2] Paris is the capital of "
                                "France.\n\nQuestion: When was the Eiffel Tower completed?"},
    {"role": "assistant", "content": "The Eiffel Tower was completed in 1889 [1]."},
    {"role": "user", "content": "Sources:\n[1] The Eiffel Tower was completed in 1889.\n[2] Paris is the capital of "
                                "France.\n\nQuestion: When was the Eiffel Tower completed in Rome?"},
    {"role": "assistant", "content": NOT_FOUND},
]


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto",
                 prompt: str = "v2") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.prompt = prompt

    def build_messages(self, question: str, sources: list[str], history: list[dict] | None = None) -> list[dict]:
        ctx = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
        if self.prompt == "v1":
            msgs = [{"role": "system", "content": SYSTEM_V1}]
        else:
            msgs = [{"role": "system", "content": SYSTEM_V2}, *FEW_SHOT]
        for h in (history or [])[-4:]:                     # short memory of the conversation
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"})
        return msgs

    @torch.inference_mode()
    def answer(self, question: str, sources: list[str], history: list[dict] | None = None,
               max_new_tokens: int = 256) -> tuple[str, list[int]]:
        """Returns (answer text, list of cited source numbers 1..n that really exist)."""
        prompt = self.tok.apply_chat_template(self.build_messages(question, sources, history),
                                              tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,   # greedy = reproducible
                                  pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(sources)})
        return text, cited
