import pandas as pd
import numpy as np
from pathlib import Path

# This is the Hugging Face datasets library (not to be confused with pytorch datasets)
from datasets import Dataset
# The transformers API from Hugging Face
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

# 'train' to fine-tune, 'load' to load an already fine-tuned model
run_mode = 'load'

current_path = Path.cwd()
data_path = current_path / 'data'
models_path = current_path / 'models'
submission_path = current_path / 'submission'

# Load the training data into a dataframe
df = pd.read_csv(data_path/'train.csv')
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

# Convert the dataframe into a Hugging Face Dataset
ds = Dataset.from_pandas(df)

# A small generally pretty good model (at least in 2022)
model_nm = 'microsoft/deberta-v3-small'

# Tokenize the input column using the Hugging Face AutoTokenizer with the model microsoft/deberta-v3-small from Hugging Face
tokz = AutoTokenizer.from_pretrained(model_nm)
def tok_func(x):
    return tokz(x["input"])
tok_ds = ds.map(tok_func, batched=True)
# Hugging Face needs the dependent variable column named as 'labels'
tok_ds = tok_ds.rename_columns({'score':'labels'})

# Transformers uses a `DatasetDict` for holding the training and validation sets.
# To create one that contains 25% of our data for the validation set, and 75% for the training set, we use `train_test_split`:
dds = tok_ds.train_test_split(0.25, seed=42)

# Load the test set. This is not the 25% validation set of the split.
# It is a completely different test set that we are not supposed to see at all while we train and test our model.
eval_df = pd.read_csv(data_path/'test.csv')
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

# Hyper parameters
# Batch size, as big as your GPU RAM can handle
bs = 128
# Run and test several epochs until the model starts to get worse
epochs = 4
# Start from a very small number, test, and start doubling until the trainer gives worse results
lr = 8e-5

args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')
# num_labels=1 converts this model to a regression model
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

# Use Pearson (r) as compute_metrics
def corr(x,y):
    return np.corrcoef(x,y)[0][1]
def corr_d(eval_pred):
    return {'pearson': corr(*eval_pred)}

if run_mode == 'train':
    # Train (fine-tune) the model using the Hugging Face AutoModelForSequenceClassification with the model microsoft/deberta-v3-small from Hugging Face
    trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'], tokenizer=tokz, compute_metrics=corr_d)
    trainer.train()
    # Save the fine tuned model and tokenizer for future use
    model.save_pretrained(models_path/'model')
    tokz.save_pretrained(models_path/'tokz')
elif run_mode == 'load':    
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(models_path/'model')
    tokz = AutoTokenizer.from_pretrained(models_path/'tokz')
    trainer = Trainer(model, args, eval_dataset=dds['test'], tokenizer=tokz, compute_metrics=corr_d)

# Use the model on the test set to get the predicted values
preds = trainer.predict(eval_ds).predictions.astype(float)
# Clip negative and >1 values
preds = np.clip(preds, 0, 1)

# Save the results into a CSV file
submission = Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})
submission.to_csv(submission_path/'submission.csv', index=False)
