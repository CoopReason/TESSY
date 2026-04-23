# Boundary Predictors Training Workflow

This directory contains scripts for automatically annotating and training Boundary Predictors.


## Training Steps

Please follow these steps to execute the training workflow:

### Step 1: Sample Reasoning  Trajectory

Sample data using both the teacher and student models.

```bash
bash script_sample_data.sh
```

### Step 2: Annotate Training Data

Using the data sampled in Step 1, we can automatically annotate style spans within them.

```bash
bash script_annotate.sh
```

### Step 3: Train Boundary Predictors
Train the Boundary Predictors using the annotated training data generated in Step 2.

```bash
bash script_train_predictor.sh
```
