import json
import argparse
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# How to run: python llm1_pipeline.py --input LLM1_input_263.json --output LLM1_output.json --model gpt-3.5-turbo

# ------------------------- Setup -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print("🔑 Loaded API Key?", OPENAI_API_KEY is not None)

file_id_cache = {}

# ------------------------- Utilities -------------------------

def load_trials(input_file):
    print(f"📂 Loading trials from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        trials = json.load(f)
    print(f"✅ Loaded {len(trials)} trials.\n")
    return trials

def extract_json(text):
    """Extract JSON object from text."""
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def upload_image_file(image_path):
    """Upload image to OpenAI Files API with purpose 'vision'."""
    if image_path in file_id_cache:
        return file_id_cache[image_path]
    with open(image_path, "rb") as f:
        result = client.files.create(file=f, purpose="vision")
    file_id_cache[image_path] = result.id
    return result.id

# ------------------------- Prompt Builder -------------------------

def build_prompt(trial):
    scenario = trial.get("scenario_json", {})
    gsr = trial.get("gsr")
    gsr_baseline = trial.get("gsr_baseline")
    images = trial.get("images", [])

    print(f"  🔹 Building prompt for trial {trial.get('trial')}...")

    scenario_text = f"Background: {scenario.get('Background','')}\nDialogue:\n"
    for k, v in scenario.items():
        if k not in ["Background", "Emotion", "Sentence", "Number"]:
            scenario_text += f"{k} {v}\n"

    if gsr:
        gsr_text = f"Physiological signals (GSR): mean={gsr.get('mean')}, max={gsr.get('max')}, min={gsr.get('min')}, std={gsr.get('std')}"
        if gsr_baseline is not None:
            gsr_text += f", baseline={gsr_baseline}"
    else:
        gsr_text = "Physiological signals (GSR): Not provided."

    prompt = f"""
You are an empathetic support chatbot. Your role is to provide emotional support to a person described in the input.
Return a SINGLE JSON object with keys:
- "arousal" (1-9)
- "valence" (1-9)
- "message" (string)
- "image_interpretation" (a list of JSONs for each image with keys: "emotion", "expression", "description")

Use **any images provided** to inform your arousal/valence predictions.

You will receive the following input fields:

- "scenario_json": conversation between two people:
    - "Your Friend_X": the USER (the person we are assessing and supporting)
    - "You_X": the USER'S FRIEND (not you, the chatbot)
    - "Background": context for the conversation
    - "Sentence" and "Number": metadata that may provide extra context
- "gsr": physiological signals of the user (mean, max, min, std), optional
- "gsr_baseline": optional baseline GSR value
- "images": facial expressions of the user, optional

Your tasks:
1. Estimate the user’s emotional **arousal** (1–9).
2. Estimate the user’s emotional **valence** (1–9).
3. Always include a "message" field with a short supportive reply.
4. Provide an "image_interpretation" list for any images.

Here is the context:

{scenario_text}

{gsr_text}

Images: {images if images else "No images."}

STRICTLY return JSON only.
"""
    print(f"  🔹 Prompt built for trial {trial['trial']}.")
    return prompt

# ------------------------- Model Call -------------------------

def call_model(model, prompt, images=None, temperature=0.7):
    input_content = [{"role": "user", "content": prompt}]

    # Determine if model supports images
    supports_images = not any(name in model.lower() for name in ["gpt-3.5", "text-davinci", "turbo-instruct"])

    if images and supports_images:
        img_inputs = [{"type": "input_text", "text": "Analyze these images and include their information in your JSON output for arousal/valence."}]
        for img_path in images:
            file_id = upload_image_file(img_path)
            print(f"      📤 Added image for {model}: {img_path}")
            img_inputs.append({"type": "input_image", "file_id": file_id})
        input_content.append({"role": "user", "content": img_inputs})

    print(f"    📨 Sending prompt to model '{model}'...")
    params = {
        "model": model,
        "input": input_content,
    }
    if model not in ["gpt-5-mini", "gpt-4.1-mini"]:
        params["temperature"] = temperature

    resp = client.responses.create(**params)
    content = resp.output_text.strip()
    parsed = extract_json(content)

    if parsed:
        scenario_output = {
            "arousal": parsed.get("arousal"),
            "valence": parsed.get("valence"),
            "message": parsed.get("message")
        }
        image_interpretation = parsed.get("image_interpretation", []) if supports_images else "Model cannot parse images"
    else:
        scenario_output = {"arousal": None, "valence": None, "message": "Could not parse model output."}
        image_interpretation = "Model cannot parse images" if images and not supports_images else "No images provided"

    print(f"  🔹 Parsing model output...")
    if scenario_output["arousal"] is not None:
        print("    ✅ JSON parsed successfully.")
    else:
        print("    ⚠ Could not parse scenario output.")

    return scenario_output, image_interpretation


# ------------------------- Main Pipeline -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="LLM1_output.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    trials = load_trials(args.input)

    results = []
    if os.path.exists(args.output):
        print(f"📂 Found existing output file '{args.output}', resuming...")
        with open(args.output, "r", encoding="utf-8") as f:
            try: results = json.load(f)
            except: pass
        print(f"✅ Loaded {len(results)} previous trials.\n")

    for trial in trials:
        print(f"▶️ Processing trial {trial['trial']} (run {trial['run']}, participant {trial['participant']})")
        start_time = time.time()

        if any(r["participant"]==trial["participant"] and r["run"]==trial["run"] and r["trial"]==trial["trial"] for r in results):
            print("  ⏩ Skipping (already processed)")
            continue

        prompt = build_prompt(trial)
        images = [Path(img).as_posix() for img in trial.get("images", [])] or None

        model_output, image_interpretation = call_model(
            model=args.model,
            prompt=prompt,
            images=images,
            temperature=args.temperature
        )

        trial_result = {
            "participant": trial["participant"],
            "run": trial["run"],
            "trial": trial["trial"],
            "model_output": model_output,
            "image_interpretation": image_interpretation
        }

        results.append(trial_result)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"⏱ Trial {trial['trial']} completed in {elapsed:.1f}s. Progress saved.\n")

    print(f"✅ All trials processed. Results in {args.output}")

if __name__ == "__main__":
    main()
