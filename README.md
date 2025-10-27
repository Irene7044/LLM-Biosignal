# LLM-Biosignal
A research on how various LLM models respond to those seeking mental health support according to their provided biosignals. 

**Note:**
Final results are recorded in files named in the format oracle_results_.json under every different model combination folder. The final summaries can be found at the bottom of the files. 
Due to the large file size, the AFFEC dataset can be found within the paper: https://arxiv.org/html/2504.18969v1
The stimulus (Scenario and facial expression images) can be accessed via the OneDrive link: https://1drv.ms/f/c/ce861069657a8bbc/El5tn2PDflRHmgoFXBAbeh8BpCDqxuqgKb0LWN7bFI6Amg?e=9CiMXl

**Files Descriptions**


**Dataset Cases**: Contains results for each model pair combination for 263 trials from the AFFEC dataset


**Extreme Cases**: Contains results for each model pair combination for 90 trials from the AFFEC dataset & LLM crafted extreme scenarios 


**data_extracting:** Contains python files for extracting data from the AFFEC dataset
- affec_loader.py = loads the data from the AFFEC dataset folders
- ground_truth_remove.py = removes the ground truth for LLM1 and LLM2 to process as inputs
- gsr_process = calculate gsr values (max, min, mean, standard deviation, baseline)


**Pipelines**

The commands for running each pipeline are written as comments at the top of each pipeline file.

- **llm1_pipeline.py** = Pipeline for empathetic support chatbot


- **llm2_supervisor.py** = Pipeline for supervisor


- **llm3_oracle_model** = Pipeline for oracle model



**Model Pair Combination Guide:**

Oracle Model is always using GPT 5.

  - Combination A1: GPT 4.1 mini (LLM1) | GPT 5 mini (LLM2)

  - Combination A2: GPT 5 mini (LLM1) | GPT 4.1 mini (LLM2)
   
  - Combination B1: GPT 3.5 turbo (LLM1) | GPT 4-o mini (LLM2)  

  - Combination B2: GPT 4-o mini (LLM1) | GPT 3.5 turbo (LLM2) 
