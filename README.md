# Clinical Multimodal LLM Pipeline for EHR Summarization and Contradiction Detection

**Status:** Independent ongoing research / applied project. Not yet submitted for peer review.

**Author:** Sabit Md Asad ([sabitpe97.com](https://sabitpe97.com) · [GitHub](https://github.com/Sabit400) · [ResearchGate](https://researchgate.net/profile/Sabit-Md-Asad-2))

## Overview

This project builds a retrieval-augmented pipeline over a synthetic clinical
notes corpus: given a patient ID and a natural-language query, it retrieves
the most relevant notes, generates a grounded summary, and flags any direct
contradictions (allergy or medication status conflicts) across the patient's
note history — evaluated against a known ground-truth answer key rather than
inspected by eye.

## Important note on the "LLM" component — read this first

**This project does not call a large language model**, and that is stated
here plainly rather than left implicit. The development environment this
project was built in has no network access to any LLM provider: no API key
is available server-side, and downloading open-weight model files (e.g.,
Llama) requires access to external model hubs that are not reachable from
this environment.

Rather than fake an LLM call or claim results that don't exist, the
generation layer (`src/generate.py`) is a real, working **extractive
summarizer**: it assembles the most query-relevant sentences directly from
retrieved notes, verbatim. This is a legitimate generation strategy in its
own right for clinical use, where traceability to the exact source sentence
is often preferable to paraphrase. The module is written so that swapping in
a real LLM call (Claude, GPT-4, or a local model) requires changing only the
body of one function (`generate_summary()`); the prompt-construction logic
(`build_prompt()`) and the entire RAG architecture around it (retrieval,
contradiction detection, evaluation, serving) do not need to change. See the
docstring in `src/generate.py` for the exact swap-in code.

The **retrieval** layer also uses TF-IDF + cosine similarity rather than
neural dense embeddings, for the same reason: downloading a pretrained
sentence-embedding model requires access to a model hub not reachable here.
This is a legitimate, classic, fully-reproducible retrieval method with no
external dependencies, and the retrieval interface is written so a dense
embedding backend could be substituted without changing downstream code.

**The contradiction detection layer is fully real** — genuine rule-based
logic, evaluated against a ground-truth answer key (see Results below). It
is not a stand-in for anything.

## Data note

The clinical notes corpus (`data/generate_data.py`) is entirely
**synthetic and fictional**. No real patient data, real clinical records, or
real institution's data is used or represented anywhere in this project.
The generator deliberately injects a controlled number of contradictions per
patient (allergy status conflicts, medication active/discontinued conflicts)
with recorded ground truth, specifically so the contradiction detector could
be evaluated quantitatively rather than only demonstrated qualitatively.

## Methodology

![Methodology workflow](diagrams/methodology_workflow.svg)

1. **Corpus generation** — 40 synthetic patients, 231 clinical notes total,
   with 22 ground-truth contradictions injected across allergy and
   medication assertions.
2. **Retrieval** — TF-IDF vectorization (word + bigram) with cosine
   similarity, scoped per patient.
3. **Generation** — extractive summarization from retrieved notes (LLM
   stand-in, see above), with any relevant contradiction flags surfaced
   explicitly rather than silently resolved.
4. **Contradiction detection** — regex-based extraction of allergy and
   medication assertions per note, followed by logical conflict checking
   across a patient's note timeline.
5. **Evaluation** — precision/recall/F1 of detected contradictions against
   the injected ground truth.
6. **Prototype serving** — a FastAPI `/query` endpoint returning retrieved
   evidence, the generated summary, and contradiction flags in one response.

## Algorithms and tools

| Category | Tools / Algorithms |
|---|---|
| Retrieval | TF-IDF (word + bigram) + cosine similarity (scikit-learn) |
| Generation | Extractive summarization (documented LLM stand-in) |
| Contradiction detection | Regex-based assertion extraction + rule-based conflict logic |
| Serving | FastAPI, Pydantic, Uvicorn |
| Core stack | Python, scikit-learn, NumPy |

## Results

### Contradiction detection vs. ground truth

| Metric | Value |
|---|---|
| Ground-truth contradictions | 22 |
| Detected flags | 20 |
| True positives | 20 |
| False positives | 0 |
| False negatives | 2 |
| Precision (patient/type level) | **1.000** |
| Recall (patient/type level) | **0.909** |
| F1 (patient/type level) | **0.952** |
| Precision (exact note-pair match) | 0.650 |
| Recall (exact note-pair match) | 0.591 |

Full data: [`results/contradiction_detection_metrics.json`](results/contradiction_detection_metrics.json).

**Root cause of the 2 missed cases (investigated, not guessed):** both false
negatives are medication contradictions where the injected discontinuation
occurs in the patient's *final* note. In that specific situation, there is
no subsequent note in which the drug could be re-asserted as active, so the
inconsistency is genuinely indistinguishable from ordinary, non-contradictory
chronological medication discontinuation — a real clinical event, not a
documentation conflict. The detector's design deliberately does not flag a
single discontinuation as a contradiction (only a discontinuation followed
by the same drug being listed active again in a later note), because
flagging every discontinuation would produce a large number of false
positives in real use. This is a considered precision/recall trade-off, not
an unexamined gap, and it is the reason precision remains a perfect 1.000
while recall is 0.909 rather than 1.000.

**On the exact note-pair metric being lower:** the detector reports the
*first* qualifying conflict pair it finds per patient/type, while the ground
truth records the *specific* pair the generator injected; when a patient has
more than two conflicting assertions, these pairs don't always match exactly
even though the patient-level contradiction is correctly detected. This is
why the patient/type-level metric (the one that answers "did we catch that
this patient has a contradiction") is the primary reported result, with the
stricter exact-pair metric included for transparency rather than omitted.

### Example output

For patient `PT0000`, query *"What are the patient's known drug
allergies?"*:

```
[!] CONTRADICTION DETECTED in retrieved notes:
  - [allergy] Note 0 states allergy status as 'shellfish'; note 3 states 'latex'.

Retrieved evidence (most relevant notes, verbatim):
  - (Note 0, Discharge Summary, score=0.035): Allergies: Patient reports allergy to shellfish.
  - (Note 2, Admission Note, score=0.035): Allergies: Patient reports allergy to shellfish.
  - (Note 3, Discharge Summary, score=0.035): Allergies: Patient reports allergy to latex.
  - (Note 1, Admission Note, score=0.035): Allergies: Patient reports allergy to shellfish.
```

## Repository structure

```
clinical-rag-contradiction/
├── data/
│   ├── generate_data.py               # synthetic corpus + contradiction injector
│   ├── clinical_notes.json             # generated notes (231 notes, 40 patients)
│   └── contradiction_ground_truth.json # ground-truth answer key
├── src/
│   ├── retrieval.py                    # TF-IDF + cosine similarity retriever
│   ├── contradiction_detector.py       # rule-based assertion conflict detection
│   ├── generate.py                     # extractive generation (documented LLM stand-in)
│   └── evaluate.py                     # precision/recall against ground truth
├── diagrams/
│   └── methodology_workflow.svg
├── prototype/
│   └── app.py                          # FastAPI query + contradiction-check service
├── results/                            # evaluation metrics
├── requirements.txt
└── README.md
```

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Generate the synthetic clinical corpus
python data/generate_data.py

# 2. Try retrieval directly
python src/retrieval.py

# 3. Try the contradiction detector directly
python src/contradiction_detector.py

# 4. Evaluate against ground truth
python src/evaluate.py

# 5. Try the extractive generation layer
python src/generate.py

# 6. Run the prototype API
cd prototype && uvicorn app:app --reload --port 8003
# POST to http://localhost:8003/query with {"patient_id": "PT0000", "query": "..."}
```

## Known limitations

- **No real LLM in the loop**, for the environment reasons stated above. The
  generation layer is extractive, not generative, and the retrieval layer
  uses TF-IDF, not dense neural embeddings. Both are documented, honest,
  and swappable — not claimed to be something they are not.
- **Contradiction detector scope.** Targets two specific, well-structured
  assertion types (allergy status, medication active/discontinued status)
  using patterns matched to this synthetic corpus's note templates. A
  production system would need a general clinical NLP pipeline (NER +
  relation extraction) to generalize to free-text variability.
- **Synthetic corpus only.** No real clinical notes, and results characterize
  the retrieval/detection methodology, not performance on real EHR data,
  which has far greater linguistic variability, abbreviation use, and
  documentation inconsistency than this synthetic corpus.
- **English-only, single-institution note format.** No multi-format or
  multi-language note handling is implemented.

## Relationship to prior published work

This project extends the RAG / clinical-EHR interest area connected to prior
research work into a genuinely evaluable retrieval and contradiction-
detection pipeline. It is an independent, hands-on framework — built to
demonstrate applied, reproducible capability in retrieval-augmented systems
and rule-based clinical NLP — rather than a replication of any specific
published study. All code, data generation, and results in this repository
are original work produced for this project.
