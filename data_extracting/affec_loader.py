import os
import json
import pandas as pd
import numpy as np
from glob import glob
from zipfile import ZipFile
import gzip

class AFFECDataLoader:
    def __init__(self, base_dir, stim2_dir, gsr_dir):
        self.base_dir = base_dir
        self.stim2_dir = stim2_dir
        self.gsr_dir = gsr_dir

    def load_participants(self):
        return [p for p in glob(os.path.join(self.base_dir, "sub-*")) if os.path.isdir(p)]

    def load_event_log(self, participant_path, run):
        """Load trial events (flags + local_time)."""
        ev_file = os.path.join(
            participant_path, f"sub-*task-fer_run-{run}_events.tsv"
        )
        matches = glob(ev_file)
        if not matches:
            raise FileNotFoundError(f"No event log for run {run} in {participant_path}")
        return pd.read_csv(matches[0], sep="\t")

    def load_annotations(self, participant_path, run):
        """Load behavioral annotations (felt/perceived)."""
        beh_file = os.path.join(
            participant_path, "beh", f"sub-*task-fer_run-{run}_beh.tsv"
        )
        matches = glob(beh_file)
        if not matches:
            raise FileNotFoundError(f"No annotations for run {run} in {participant_path}")
        return pd.read_csv(matches[0], sep="\t")

    def find_stimulus_files(self, stim_file):
        """Find matching scenario JSON (loaded) + image paths."""
        stem = os.path.splitext(os.path.basename(stim_file))[0]
        matches = glob(os.path.join(self.stim2_dir, "**", f"*{stem}*"), recursive=True)

        # JSON (load contents if found)
        scenario_json = None
        json_matches = [m for m in matches if m.endswith(".json")]
        if json_matches:
            with open(json_matches[0], "r", encoding="utf-8") as f:
                try:
                    scenario_json = json.load(f)  # load as dict
                except json.JSONDecodeError:
                    scenario_json = f.read()      # fallback: keep as raw text

        # Images (filter out scenario_*)
        images = [
            m for m in matches
            if m.endswith((".jpg", ".jpeg", ".png"))
            and not os.path.basename(m).startswith("scenario_")
        ]

        return {
            "json": scenario_json,  # <- now actual contents, not path
            "images": images        # keep paths to images
        }

    def load_gsr(self, participant_id, run, onset_sec, offset_sec, step=3):
        """Extract GSR data for the trial time frame, every `step` samples."""
        gsr_path = os.path.join(self.gsr_dir, participant_id, "beh")
        file_name = f"{participant_id}_task-fer_run-{run}_recording-gsr_physio.tsv.gz"
        gsr_file = os.path.join(gsr_path, file_name)
        print(f"[DEBUG] Looking for GSR TSV at: {gsr_file}")

        if not os.path.exists(gsr_file):
            print(f"[WARN] GSR TSV file not found for {participant_id}, run {run}")
            return None

        # Open the gzip TSV
        with gzip.open(gsr_file, "rt") as f:
            gsr = pd.read_csv(f, sep="\t", header=None)
        
        print(f"[DEBUG] Loaded TSV: {gsr.shape[0]} rows, {gsr.shape[1]} columns")

        # Use first column as onset (seconds), last column as GSR
        gsr.columns = ["onset"] + [f"col{i}" for i in range(1, gsr.shape[1]-1)] + ["GSR_Conductance_cal"]

        # Filter rows within trial time frame
        segment_full = gsr[(gsr["onset"] >= onset_sec) & (gsr["onset"] <= offset_sec)]
        
        # Take every `step`-th row
        segment = segment_full.iloc[::step].reset_index(drop=True)

        if segment.empty:
            print(f"[WARN] No GSR data found within trial time frame for {participant_id}, run {run}")
            return None

        # Print the actual GSR onset range for the selected rows
        print(f"[DEBUG] Trial time frame: {onset_sec}-{offset_sec}, "
            f"selected segment GSR onset range: {segment['onset'].min()}-{segment['onset'].max()}, "
            f"selected rows: {len(segment)}")

        return segment[["onset", "GSR_Conductance_cal"]]

    def assemble_trials(self, participant_path):
        participant_id = os.path.basename(participant_path)
        print(f"Processing participant: {participant_id}")

        trials = []

        for run in range(4):
            print(f"  Processing run: {run}")
            try:
                events = self.load_event_log(participant_path, run)
                annots = self.load_annotations(participant_path, run)
            except FileNotFoundError:
                print(f"    Skipping run {run} (files not found)")
                continue

            for _, row in annots.iterrows():
                trial_num = row["trial"]
                print(f"    Processing trial: {trial_num}")

                trial_events = events[events["trial"] == trial_num]

                # Use onset/offset in seconds from the first "scenario" and last "last_frame_video"
                onset_rows = trial_events[trial_events["flag"] == "scenario"]
                offset_rows = trial_events[trial_events["flag"] == "last_frame_video"]

                if onset_rows.empty or offset_rows.empty:
                    print(f"      Skipping trial {trial_num} (missing onset/offset flags)")
                    continue

                # These columns are in seconds (replace 'local_time' with the actual onset column)
                onset = onset_rows.iloc[0]["onset"]   # in seconds
                offset = offset_rows.iloc[-1]["onset"]  # end time

                print(f"[DEBUG] Trial time frame: {onset}-{offset}")

                # Stimulus
                stim_data = self.find_stimulus_files(row["stim_file"])

                # GSR segment for the trial
                gsr_segment = self.load_gsr(participant_id, run, onset, offset, step=3)

                # Baseline: find onset of "first_fix"
                baseline_value = None
                first_fix_rows = trial_events[trial_events["flag"] == "first_fix"]
                if not first_fix_rows.empty and gsr_segment is not None:
                    first_fix_onset = first_fix_rows.iloc[0]["onset"]
                    # Find closest GSR row to first_fix_onset
                    idx = (gsr_segment["onset"] - first_fix_onset).abs().idxmin()
                    baseline_value = float(gsr_segment.loc[idx, "GSR_Conductance_cal"])

                trials.append({
                    "participant": participant_id,
                    "run": run,
                    "trial": trial_num,
                    "trial_type": row.get("trial_type", None),
                    "scenario_json": stim_data["json"],
                    "images": stim_data["images"],
                    "gsr": gsr_segment,
                    "gsr_baseline": baseline_value,   # ✅ new field
                    "f_emotion": {"valence": row["f_emotion_v"], "arousal": row["f_emotion_a"]},
                    "p_emotion": {"valence": row["p_emotion_v"], "arousal": row["p_emotion_a"]}
                })

                if gsr_segment is not None:
                    print(f"[DEBUG] Selected GSR rows: {len(gsr_segment)}")
                else:
                    print(f"[WARN] No GSR data found for trial {trial_num}")

        print(f"Finished participant: {participant_id}, loaded {len(trials)} trials\n")
        return trials

    def build_dataset(self):
        dataset = []
        for p in self.load_participants():
            dataset.extend(self.assemble_trials(p))
        return dataset


# Main

if __name__ == "__main__":
    loader = AFFECDataLoader(
        base_dir=r"C:\Users\Irene\OneDrive\Desktop\Adelaide UNI\Year 2\Sem 2\Topics\Dataset\core",
        stim2_dir=r"C:\Users\Irene\OneDrive\Desktop\Adelaide UNI\Year 2\Sem 2\Topics\Dataset\stim 2\stim\data",
        gsr_dir=r"C:\Users\Irene\OneDrive\Desktop\Adelaide UNI\Year 2\Sem 2\Topics\Dataset\gsr"
    )

    json_file = "affec_dataset.json"

    if os.path.exists(json_file):
        print(f"Loading dataset from {json_file}...")
        with open(json_file, "r") as f:
            dataset_serializable = json.load(f)

        # Convert GSR back to DataFrame
        for trial in dataset_serializable:
            if trial["gsr"] is not None:
                trial["gsr"] = pd.DataFrame(trial["gsr"])

        dataset = dataset_serializable
        print(f"Loaded {len(dataset)} trials from JSON.")

    else:
        print("JSON file not found. Building dataset from raw files...")
        dataset = loader.build_dataset()
        print(f"Loaded {len(dataset)} trials from raw files.")

        # Convert GSR DataFrames to JSON-serializable format
        dataset_serializable = []
        for trial in dataset:
            trial_copy = trial.copy()
            if trial_copy["gsr"] is not None:
                trial_copy["gsr"] = trial_copy["gsr"].to_dict(orient="records")
            dataset_serializable.append(trial_copy)

        # Save to JSON
        with open(json_file, "w") as f:
            json.dump(dataset_serializable, f, indent=2)
        print(f"Dataset saved to {json_file}.")
