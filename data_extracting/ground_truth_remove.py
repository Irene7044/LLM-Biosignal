import json

def clean_dataset(json_input_path, json_output_path):
    """
    Removes the following fields from each trial:
    - 'trial_type'
    - 'p_emotion'
    - 'Emotion' inside 'scenario_json'
    Saves the cleaned dataset to a new JSON file.
    """
    with open(json_input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    updated_data = []
    for trial in dataset:
        trial_copy = trial.copy()

        # Remove top-level fields
        trial_copy.pop("trial_type", None)
        trial_copy.pop("p_emotion", None)

        # Remove 'Emotion' inside scenario_json if it exists
        if "scenario_json" in trial_copy and "Emotion" in trial_copy["scenario_json"]:
            trial_copy["scenario_json"].pop("Emotion")

        updated_data.append(trial_copy)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned dataset saved to {json_output_path}")


# Example usage:
clean_dataset("affec_dataset_processed.json", "LLM1_input_all.json")
