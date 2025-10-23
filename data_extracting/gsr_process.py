import json
import pandas as pd

def summarize_gsr_only(json_input_path, json_output_path):
    """
    Reads the dataset JSON, calculates mean, max, min, std of GSR
    for each trial, replaces the 'gsr' field with stats,
    and preserves 'gsr_baseline' if it exists.
    """
    with open(json_input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    summarized_data = []

    for trial in dataset:
        gsr_data = trial.get("gsr")
        stats = {}
        if gsr_data:
            df = pd.DataFrame(gsr_data)
            stats = {
                "mean": df["GSR_Conductance_cal"].mean(),
                "max": df["GSR_Conductance_cal"].max(),
                "min": df["GSR_Conductance_cal"].min(),
                "std": df["GSR_Conductance_cal"].std()
            }
        else:
            stats = {"mean": None, "max": None, "min": None, "std": None}

        trial_copy = trial.copy()
        trial_copy["gsr"] = stats  # replace gsr with summary stats

        # ✅ Preserve gsr_baseline if it exists
        if "gsr_baseline" in trial:
            trial_copy["gsr_baseline"] = trial["gsr_baseline"]

        summarized_data.append(trial_copy)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(summarized_data, f, indent=2)

    print(f"✅ Saved summarized GSR dataset to {json_output_path}")


def remove_f_emotion(json_input_path, json_output_path):
    """
    Removes 'f_emotion' field from each trial in the dataset JSON
    and saves to a new JSON file.
    """
    with open(json_input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    updated_data = []
    for trial in dataset:
        trial_copy = trial.copy()
        trial_copy.pop("f_emotion", None)  # safe remove
        updated_data.append(trial_copy)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2)

    print(f"✅ 'f_emotion' removed. Saved updated dataset to {json_output_path}")


# Example usage:
summarize_gsr_only("affec_dataset.json", "affec_dataset_gsr_processed.json")
remove_f_emotion("affec_dataset_gsr_processed.json", "affec_dataset_femotion_removed.json")
