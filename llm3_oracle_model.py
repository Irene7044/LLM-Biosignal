#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import ollama
import numpy as np

with open("llm3_prompt.txt", encoding="utf-8") as f:
    GUIDELINES = f.read()


# Running: python llm3_oracle_model.py --inputs oracle_input_ES.json --llm-out LLM2_output_GPT3.5turbo.json --out oracle_results_B2.json --backend openai --model gpt-5


# ------------------------- Setup -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print("🔑 Loaded API Key?", OPENAI_API_KEY is not None)

# ------------------------- Utilities -------------------------
def load_json(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def is_accurate(pred, gt, tol=1):
    if pred is None or gt is None:
        return None
    return abs(pred - gt) <= tol

def call_llm(backend, model, prompt):
    """Call either OpenAI or Ollama backend with minimal token usage."""
    if backend == "openai":
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        return resp.choices[0].message.content.strip()
    elif backend == "ollama":
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp["message"]["content"].strip()
    else:
        raise ValueError(f"Unknown backend: {backend}")

# ------------------------- Main Oracle Pipeline -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--llm-out", required=True)
    parser.add_argument("--out", default="oracle_output.json")
    parser.add_argument("--backend", choices=["openai", "ollama", "none"], default="none")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    inputs = load_json(args.inputs)
    llm_out_raw = load_json(args.llm_out)
    llm_out = llm_out_raw.get("trials", []) if isinstance(llm_out_raw, dict) else llm_out_raw

    results = []
    if Path(args.out).exists():
        print(f"📂 Found existing output file '{args.out}', resuming...")
        prev = load_json(args.out)
        results = prev.get("trials", []) if isinstance(prev, dict) else prev
        print(f"✅ Loaded {len(results)} previous trials.\n")

    done_keys = {f"{r['participant']}-{r['run']}-{r['trial']}" for r in results}
    llm_by_key = {f"{row.get('participant','NA')}-{row.get('run','NA')}-{row.get('trial','NA')}": row for row in llm_out}

    for trial in inputs:
        pid = trial.get("participant", "NA")
        run = trial.get("run", "NA")
        tnum = trial.get("trial", "NA")
        key = f"{pid}-{run}-{tnum}"

        if key in done_keys:
            continue

        print(f"▶️ Processing trial {tnum} (participant {pid}, run {run})")
        start_time = time.time()

        llm_row = llm_by_key.get(key)
        if not llm_row:
            results.append({"participant": pid, "run": run, "trial": tnum, "error": "Missing LLM outputs"})
            continue

        gt_arousal = safe_float(trial.get("p_emotion", {}).get("arousal"))
        gt_valence = safe_float(trial.get("p_emotion", {}).get("valence"))
        gt_emotion = trial.get("scenario_json", {}).get("Emotion")

        llm1 = llm_row.get("llm1", {})
        llm2 = llm_row.get("llm2", {})

        llm1_ar = safe_float(llm1.get("arousal"))
        llm1_va = safe_float(llm1.get("valence"))
        llm1_msg = llm1.get("message", "")

        llm2_ar = safe_float(llm2.get("arousal"))
        llm2_va = safe_float(llm2.get("valence"))
        llm2_decision = llm2.get("decision")
        llm2_judgement = llm_row.get("judgement", {}).get("llm1_response_quality", "appropriate")

        # ------------------------- Prediction Evaluation -------------------------
        prediction_eval = {
            "llm1": {
                "arousal": {
                    "pred": llm1_ar,
                    "true": gt_arousal,
                    "error": abs(llm1_ar - gt_arousal) if (llm1_ar is not None and gt_arousal is not None) else None,
                    "assessment": "Accurate" if is_accurate(llm1_ar, gt_arousal) else "Under/Overestimated"
                },
                "valence": {
                    "pred": llm1_va,
                    "true": gt_valence,
                    "error": abs(llm1_va - gt_valence) if (llm1_va is not None and gt_valence is not None) else None,
                    "assessment": "Accurate" if is_accurate(llm1_va, gt_valence) else "Under/Overestimated"
                }
            },
            "llm2": {
                "arousal": {
                    "pred": llm2_ar,
                    "true": gt_arousal,
                    "error": abs(llm2_ar - gt_arousal) if (llm2_ar is not None and gt_arousal is not None) else None,
                    "assessment": "Accurate" if is_accurate(llm2_ar, gt_arousal) else "Under/Overestimated"
                },
                "valence": {
                    "pred": llm2_va,
                    "true": gt_valence,
                    "error": abs(llm2_va - gt_valence) if (llm2_va is not None and gt_valence is not None) else None,
                    "assessment": "Accurate" if is_accurate(llm2_va, gt_valence) else "Under/Overestimated"
                }
            }
        }

        # ------------------------- Oracle Evaluations -------------------------
        llm1_eval_text = None
        llm2_eval_text = None
        oracle_summary = None

        if args.backend != "none":
            llm1_image_desc = llm1.get("image_interpretation", [])
            # Handle older models that return string for image interpretation
            if llm1_image_desc:
                if isinstance(llm1_image_desc, list) and isinstance(llm1_image_desc[0], dict):
                    image_summary = llm1_image_desc[0]['emotion'] + ", " + llm1_image_desc[0]['expression']
                else:
                    image_summary = str(llm1_image_desc)
            else:
                image_summary = "No image data"


            scenario_sentence = trial.get("scenario_json", {}).get("Sentence", "")
            scenario_background = trial.get("scenario_json", {}).get("Background", "")

            # ---------- MINIMAL PROMPT PER TRIAL ----------
            prompt = f"""
            You are an evaluator for an empathetic chatbot (LLM1) and supervisor (LLM2). 
            Scenario: {scenario_sentence}, Background: {scenario_background}
            Ground truth: Arousal={gt_arousal}, Valence={gt_valence}, Emotion={gt_emotion}
            LLM1 response: {llm1_msg}, Predicted arousal={llm1_ar}, valence={llm1_va}
            LLM2 judgement: {llm2_judgement}, Decision: {llm2_decision}, Predicted arousal={llm2_ar}, valence={llm2_va}
            Image summary: {image_summary}

            Evaluate LLM1 and LLM2 response for appropriateness and correctness.
            Return STRICT JSON only with keys: "llm1_eval", "llm2_eval", "summary"
            """

            oracle_output = call_llm(args.backend, args.model, prompt)

        try:
            oracle_json = json.loads(oracle_output)
            llm1_eval_text = oracle_json.get("llm1_eval", "No evaluation")
            llm2_eval_text = oracle_json.get("llm2_eval", "No evaluation")
            oracle_summary = oracle_json.get("summary", "No summary")
        except json.JSONDecodeError:
            print("⚠️ Oracle output was not valid JSON, storing raw text instead")
            llm1_eval_text = oracle_output
            llm2_eval_text = oracle_output
            oracle_summary = oracle_output


        results.append({
            "participant": pid,
            "run": run,
            "trial": tnum,
            "ground_truth": {"arousal": gt_arousal, "valence": gt_valence, "emotion": gt_emotion},
            "llm1": {
                "index": 1,
                "prediction": {
                    "response": llm1_msg  # only keep response, remove arousal/valence here
                },
                "llm1_eval": llm1_eval_text
            },
            "llm2": {
                "index": 2,
                "prediction": {
                    "decision": llm2_decision  # only keep decision, remove arousal/valence here
                },
                "judgement": llm2_judgement,
                "llm2_eval": llm2_eval_text
            },
            "prediction_eval": prediction_eval,  # keep full arousal/valence prediction evaluation
            "oracle": {"index": 3, "summary": oracle_summary}
        })


        save_json(args.out, {"trials": results})  # save after each trial
        elapsed = time.time() - start_time
        print(f"⏱ Trial {tnum} completed in {elapsed:.1f}s. Progress saved.\n")

    # ------------------------- Final Analysis -------------------------

    def _as_text(x):
        """Return a lowercased string representation for x (handles str/dict/None)."""
        if isinstance(x, str):
            return x.lower()
        if x is None:
            return ""
        try:
            return json.dumps(x).lower()
        except Exception:
            return str(x).lower()


    total_trials = len(results)

    llm1_ar_correct = sum(1 for r in results if r.get("prediction_eval", {}).get("llm1", {}).get("arousal", {}).get("assessment") == "Accurate")
    llm1_va_correct = sum(1 for r in results if r.get("prediction_eval", {}).get("llm1", {}).get("valence", {}).get("assessment") == "Accurate")
    llm2_ar_correct = sum(1 for r in results if r.get("prediction_eval", {}).get("llm2", {}).get("arousal", {}).get("assessment") == "Accurate")
    llm2_va_correct = sum(1 for r in results if r.get("prediction_eval", {}).get("llm2", {}).get("valence", {}).get("assessment") == "Accurate")

    llm1_appropriate_count = sum(
        1 for r in results
        if "appropriate" in _as_text(r.get("llm1", {}).get("llm1_eval", ""))
    )

    llm1_inappropriate_count = total_trials - llm1_appropriate_count

    llm2_correct_judgement_count = sum(
        1 for r in results
        if "correct" in _as_text(r.get("llm2", {}).get("llm2_eval", ""))
    )
    
    llm2_incorrect_judgement_count = total_trials - llm2_correct_judgement_count

    final_analysis = {
        "total_trials": total_trials,
        "llm1_accuracy": {
            "arousal": f"{round((llm1_ar_correct / total_trials * 100), 2)}%" if total_trials else None,
            "valence": f"{round((llm1_va_correct / total_trials * 100), 2)}%" if total_trials else None,
            "appropriate_count": llm1_appropriate_count,
            "appropriate": f"{round((llm1_appropriate_count / total_trials * 100), 2)}%" if total_trials else None,
            "inappropriate_count": llm1_inappropriate_count,
            "inappropriate": f"{round((llm1_inappropriate_count / total_trials * 100), 2)}%" if total_trials else None
        },
        "llm2_accuracy": {
            "arousal": f"{round((llm2_ar_correct / total_trials * 100), 2)}%" if total_trials else None,
            "valence": f"{round((llm2_va_correct / total_trials * 100), 2)}%" if total_trials else None,
            "correct_judgement_count": llm2_correct_judgement_count,
            "correct_judgement": f"{round((llm2_correct_judgement_count / total_trials * 100), 2)}%" if total_trials else None,
            "incorrect_judgement_count": llm2_incorrect_judgement_count,
            "incorrect_judgement": f"{round((llm2_incorrect_judgement_count / total_trials * 100), 2)}%" if total_trials else None
        }
    }

        # ------------------------- LLM3 Summary Analysis -------------------------
    valid_trials = [r for r in results if r.get("prediction_eval") is not None]

    if valid_trials and args.backend != "none":
        summary_prompt = (
            "You are an expert analyst of LLM predictions in an empathetic chatbot study. "
            "Given the following processed trial data, summarize:\n"
            "1. Which LLM (LLM1 or LLM2) generally has higher prediction accuracy for arousal and valence, and why.\n"
            "2. Situations where LLM1 usually makes an inappropriate response.\n"
            "3. Situations where LLM2 usually makes incorrect decisions (defer the conversation to professional human counsellor or respond).\n"
            "4. Situations where LLM2 usually makes incorrect judgements of LLM1's responses. \n"
            "5. Any other notable patterns or insights.\n\n"
            f"Processed trial data (ignore trials with null results):\n{json.dumps(valid_trials, indent=2)}\n\n"
            "Provide the summary in a concise JSON-friendly text."
        )

        llm3_summary = call_llm(args.backend, args.model, summary_prompt)
        final_analysis["llm3_summary"] = llm3_summary
    else:
        final_analysis["llm3_summary"] = "No valid trials or backend set to 'none'; summary not generated."


    save_json(args.out, {"trials": results, "final_analysis": final_analysis})
    print(f"✅ All trials processed. Final results saved to {args.out}")

if __name__ == "__main__":
    main()
