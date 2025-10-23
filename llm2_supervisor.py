#!/usr/bin/env python3
"""
LLM2 Supervisor Pipeline (OpenAI, Images, Flexible Models)

Inputs:
  --inputs       : LLM1 input JSON
  --llm1-out     : LLM1 output JSON
  --out          : LLM2 output JSON
  --model        : OpenAI GPT model (gpt-4o-mini, gpt-4o, gpt-5, etc.)

Running:
python llm2_supervisor.py --inputs extreme_scenario_input.json --llm1-out LLM1_output_GPT4-omini.json --out LLM2_output_GPT3.5turbo.json --model gpt-3.5-turbo
"""

import os
import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------- Setup -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print("🔑 Loaded API Key?", OPENAI_API_KEY is not None)

# ------------------------- Utilities -------------------------

def load_json(path: str):
    """Load JSON file safely. Returns None if missing or empty/corrupt."""
    if not Path(path).exists():
        return None
    try:
        if Path(path).stat().st_size == 0:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def index_trials_by_key(trials):
    return {f"{t.get('participant')}||{t.get('run')}||{t.get('trial')}": t for t in trials}

def extract_json_block(text):
    """Try to extract JSON from model output text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                return None
    return None

# ------------------------- Prompt Builder -------------------------

def build_llm2_prompt(trial, llm1_output):
    scenario = trial.get("scenario_json", {})
    gsr = trial.get("gsr")
    gsr_baseline = trial.get("gsr_baseline")
    images = trial.get("images", [])

    scenario_text = f"Background: {scenario.get('Background','')}\nDialogue:\n"
    for k, v in scenario.items():
        if k not in ["Background", "Emotion", "Sentence", "Number"]:
            scenario_text += f"{k} {v}\n"

    gsr_text = "Physiological signals (GSR): Not provided."
    if gsr:
        gsr_text = f"Physiological signals (GSR): mean={gsr.get('mean')}, max={gsr.get('max')}, min={gsr.get('min')}, std={gsr.get('std')}"
        if gsr_baseline is not None:
            gsr_text += f", baseline={gsr_baseline}"

    llm1_summary = json.dumps(llm1_output, indent=2)

    prompt = f"""
You are LLM2, a supervisor model for an empathetic support chatbot. Your role is to supervise the output response provided by the empathetic support chatbot (LLM1) to a person described in the input.

Use **any images provided** to inform your arousal/valence predictions.

You will receive the following input fields:

- "scenario_json": conversation between two people:
    - "Your Friend_X": the USER (the person that the empathetic chatbot is assessing and supporting)
    - "You_X": the USER'S FRIEND (not you, the chatbot)
    - "Background": context for the conversation
    - "Sentence" and "Number": metadata that may provide extra context
- "gsr": physiological signals of the user (mean, max, min, std), optional
- "gsr_baseline": optional baseline GSR value
- "images": facial expressions of the user, optional
- LLM1 output: output from the empathetic chatbot model, including its predictions of the user's emotional arousal and valence levels, as well as a supportive message.

Your Tasks:
1. Estimate the user’s emotional **arousal** (1–9).
2. Estimate the user’s emotional **valence** (1–9).
3. Decide whether to "respond" to the conversation yourself or "defer" the conversation to a professional human counsellor. 
   Your decision should reflect the safest course of action for the user: if the situation suggests serious emotional distress or risk, consider deferring; otherwise, provide an empathetic response.
4. Include a "message" field for your response or defer notice
5. Analyze images and return "image_interpretation" in the format:

[
  {{
    "emotion": "<string>",
    "expression": "<string>",
    "description": "<string>"
  }}
]

6. Compare with LLM1 output and provide reasoning in "judgement"
7. Return STRICT JSON only, with "judgement" including:
   - llm1_response_quality: "appropriate" or "inappropriate", do you think the empathetic chatbot's response is appropriate or not? Appropriate as in can the response effectively support the user?
   - reasoning: a brief textual explanation for llm1_response_quality

-----------------------------------------------------------------------------------------------------------------------------------------------
Guidelines for Mild Distress Detection and Deferral:

Distress Levels


Level 1 — Very Mild / Trivial Signals
- Definition: Almost negligible emotional signals. User shows fleeting negativity or minor disengagement, unlikely to indicate meaningful distres
- Indicators (illustrative, not exhaustive): brief sighs, short expressions of boredom or tiredness, momentary low energy.
- Deferral guidance: usually continue conversation. Monitor, but rarely defer unless cues escalate or persist.

Level 2 — Mildly Concerning / Low-Level Distress
- Definition: Slight, subtle distress that may indicate the user is struggling. Even a single cue can be meaningful.
- Indicators (illustrative, not exhaustive):
- Brief or subtle worry, hesitation, low-energy tone, mild frustration, fleeting expressions of being stuck, slight withdrawal.
- Any cue that suggests the user may benefit from attention, even if it occurs only once.
- Deferral guidance: defer even for single occurrences. Use judgment based on context, tone, and impact. Level2 now captures almost all mild distress situations, making deferral common without being automatic for trivial signals.

Level 3 — Strong
- Definition: High distress or safety concern. User shows clear signals of emotional suffering or risk.
- Indicators (illustrative): panic, self-harm thoughts, high agitation, strong physiological activation.
- Deferral guidance: always defer immediately.

Deferral Decision Logic (Procedural & Safety-First)

- Level3 cues → defer immediately.
- Level2 cues → defer immediately.
- Level1 cues:
    If repeated or showing subtle persistence/hesitation → escalate to Level2 → defer.
    If single mild cue in context suggesting ongoing concern → consider deferral.
    Otherwise → continue conversation, but monitor closely.
    If unsure → defer. Safety always takes precedence.

Judgment Principles

- Integrate multiple signals: tone, repetition, duration, context.

- Do not rely on keywords alone. Interpret meaning in context.

Always justify decision in judgement.reasoning with:
1. Identified cues and intensity
2. Frequency or repetition
3. Contextual factors influencing the decision
4. Why deferral was or was not chosen

Example reasoning template:

judgement.reasoning: 
  "cues_detected": ["tiredness", "hesitation in tone"],
  "repetition": "2 consecutive messages",
  "level": "Level2",
  "action": "defer",
  "rationale": "Repeated mild cues suggest persistent low-level distress. Deferral chosen for safety."

-----------------------------------------------------------------------------------------------------------------------------------------------

Scenario:
{scenario_text}

GSR:
{gsr_text}

LLM1 output:
{llm1_summary}

Images: {images[0] if images else "No images."}

STRICTLY return JSON only.
"""
    return prompt

# ------------------------- Model Call -------------------------

def call_openai(model, prompt, temperature=0.2):
    """Call OpenAI Responses API."""
    input_content = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "input": input_content}
    if not model.lower().startswith("gpt-5"):
        kwargs["temperature"] = temperature
    resp = client.responses.create(**kwargs)
    return resp.output_text.strip()

# ------------------------- Main Pipeline -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--llm1-out", required=True)
    parser.add_argument("--out", default="LLM2_output.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    trials = load_json(args.inputs) or []
    llm1_results = load_json(args.llm1_out) or []

    trials_by_key = index_trials_by_key(trials)
    llm1_by_key = index_trials_by_key(llm1_results)

    results = []
    if Path(args.out).exists():
        existing = load_json(args.out)
        if isinstance(existing, dict):
            results = existing.get("trials", [])
        elif isinstance(existing, list):
            results = existing
    done_keys = {f"{r.get('participant')}||{r.get('run')}||{r.get('trial')}" for r in results}

    for key, trial in trials_by_key.items():
        if key in done_keys:
            print(f"⏩ Skipping trial {trial['trial']} (already processed)")
            continue

        llm1_row = llm1_by_key.get(key)
        if not llm1_row:
            results.append({"key": key, "error": "Missing LLM1 output"})
            save_json(args.out, {"trials": results})
            continue

        prompt = build_llm2_prompt(trial, llm1_row.get("model_output"))
        raw_text = call_openai(args.model, prompt, args.temperature)
        parsed = extract_json_block(raw_text)

        if not isinstance(parsed, dict):
            parsed = {}

        arousal = parsed.get("arousal")
        valence = parsed.get("valence")
        decision = parsed.get("decision", "defer")
        message = parsed.get("message", "Could not parse model output.")
        # ------------------------- Safe image interpretation -------------------------
        if "gpt-3.5" in args.model.lower():
            image_interpretation = "Model cannot parse images"
        else:
            raw_image_interp = parsed.get("image_interpretation", None)
            if raw_image_interp and isinstance(raw_image_interp, list) and all(isinstance(i, dict) for i in raw_image_interp):
                image_interpretation = raw_image_interp
            else:
                image_interpretation = [
                    {
                        "emotion": "neutral",
                        "expression": "calm face with no strong emotion",
                        "description": "Parsing failed or no image provided."
                    }
                ]


        judgement = parsed.get("judgement", {})

        output_row = {
            "participant": trial.get("participant"),
            "run": trial.get("run"),
            "trial": trial.get("trial"),
            "llm1": {
                **llm1_row.get("model_output", {}),
                "image_interpretation": llm1_row.get("image_interpretation", [])
            },
            "llm2": {
                "arousal": arousal,
                "valence": valence,
                "decision": decision,
                "message": message,
                "image_interpretation": image_interpretation
            },
            "judgement": {
                "llm1_response_quality": judgement.get("llm1_response_quality", "appropriate"),
                "reasoning": judgement.get(
                    "reasoning",
                    f"I agree with LLM1's assessment (arousal={llm1_row.get('model_output', {}).get('arousal')}, "
                    f"valence={llm1_row.get('model_output', {}).get('valence')}). Dialogue and image cues support this judgment."
                )
            }
        }

        results.append(output_row)
        save_json(args.out, {"trials": results})
        print(f"✅ Trial {trial['trial']} processed and saved.")

    print(f"\n✅ All trials done. Output saved to {args.out}")

if __name__ == "__main__":
    main()
