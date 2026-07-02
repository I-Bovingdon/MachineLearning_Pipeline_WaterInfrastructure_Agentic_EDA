# Part 3 of 3 — Interactive Annexes

`03_interactive_annexes.ipynb` · Pipeline position: **`01 → 02 → [ 03 ]`**

The productisation layer: three local web interfaces that turn the pipeline's results into tools usable by non-technical stakeholders — an agentic EDA analyst, a live prediction demo, and a conversational assistant grounded in the actual analysis.

**Prerequisites:** run notebooks 01 and 02 first. Annexes A and C additionally require an Anthropic API key in a `.env` file next to the notebook:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## The three annexes

| Annex | URL | API key | What it does |
|---|---|---|---|
| **A — LangGraph EDA Agent** | `http://127.0.0.1:7861` | Yes | Agentic pipeline that walks the dataset in 4 graph nodes and answers questions about its findings |
| **B — Data Explorer & Predictor** | `http://127.0.0.1:7860` | No | Interactive EDA + live single-pump prediction with class probabilities |
| **C — "Ask the Project"** | `http://127.0.0.1:7862` | Yes | Natural-language assistant that knows the data, decisions and results |

### Annex A — EDA Analysis Agent (LangGraph + Claude)

LangGraph orchestrates an agent as a state graph of four sequential nodes:

```
analyse_structure → analyse_quality → analyse_distributions → generate_summary
```

Each node enriches the state with findings; Claude generates the conclusions. The result is served with an interactive chat so the report can be interrogated in natural language.

### Annex B — Data Explorer & Individual Predictor (Gradio)

![Annex B](images/EDA&Predictor.png)

Two tabs, no code required:
- **EDA Explorer** — pick any categorical variable and instantly see the pump-status distribution
- **Individual prediction** — enter a pump's characteristics and get a live prediction with per-class probabilities

The predictor transforms the input with the **exact same `feature_pipeline.py`** module generated in notebook 01, and applies the tuned decision threshold (`BEST_THR` from notebook 02) — so the demo's predictions are byte-identical to the pipeline's.

### Annex C — "Ask the Project" AI Assistant

![Annex C](images/LLM%20Assistant.png)

A conversational interface (Claude via the Anthropic API) grounded in the project's real context: dataset size, issues found, strategies tested, per-region failure rates and final metrics. Error messages are translated into plain language, and each turn is labelled **Request** / **Answer**.

---

## Inputs (produced by notebooks 01 and 02)

```
artifacts/feature_artifacts.pkl    # fitted pipeline parameters
artifacts/X_eng.parquet            # column order + medians for input alignment
artifacts/rf_final.joblib          # final trained model
artifacts/model_config.json        # tuned decision threshold (BEST_THR)
feature_pipeline.py                # transformation module (imported)
```

## How to run

```bash
pip install gradio langgraph langchain-anthropic anthropic python-dotenv pyarrow joblib
```

1. Run notebooks 01 and 02 first
2. Create `.env` with your `ANTHROPIC_API_KEY` (Annex B works without it)
3. `Kernel → Restart & Run All` — the three servers stay alive while the kernel runs

> Without an API key, Annexes A and C print a clear notice and skip startup instead of failing — Annex B remains fully functional.
