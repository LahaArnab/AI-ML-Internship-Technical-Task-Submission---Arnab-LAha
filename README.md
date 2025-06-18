# AI-ML-Internship-Technical-Task-Submission---Arnab-Laha

# Command Line Q&A Model Training Report

## Dataset Overview

- **Source Data**: Command line Q&A dataset containing 200 entries
- **Categories**: git, bash, tar, grep, python-venv (40 entries each)
- **Format**: Questions and answers in CSV and JSON formats
- **Data Fields**: 
  - Original: question, command, category, tag, votes, answer_votes
  - Processed: question, answer

## Data Preparation

1. **Data Cleaning**
   - Removed unnecessary columns (category, tag, votes, answer_votes)
   - Renamed 'command' column to 'answer'
   - Final dataset shape: (200, 2)

2. **Format Conversion**
   - CSV to JSON conversion for model compatibility
   - Preserved question-answer pairs structure
   - Saved in both formats for flexibility

## Model Training

1. **Model Architecture**
   - Base Model: TinyLlama-1.1B-Chat-v1.0
   - Training Method: Supervised Fine-Tuning (SFT)
   - lightning.ai T4 GPU Training Configuration

2. **Training Configuration**
   - Batch Size: 4 
   - Gradient Accumulation Steps: 8
   - Learning Rate: 2e-4
   - Scheduler: Cosine
   - Training Epochs: 65
   - Max Steps: 250

3. **PEFT Configuration**
   - Method: LoRA
   - r: 8
   - alpha: 16
   - dropout: 0.05
   - task type: CAUSAL_LM

## File Structure
```
final/
├── data/
│   ├── final_dataset.csv
│   └── final_dataset.json
├── Training notebook/
│   └── TinyLlama_fine_tuning.ipynb
└── model/
    └── tinyllama-colorist-v1/
        └── checkpoint-250/
```


