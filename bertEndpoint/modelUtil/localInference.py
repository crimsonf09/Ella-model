import fastapi
import pandas as pd
import uvicorn
from fastapi import Request, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modelUtil.bert import BertClassifier
from datasets import Dataset
import numpy as np
from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel
from typing import Optional

type2ind = {}
ind2type = {}

type = [
    "research_insight",
    "strategy_planning",
    "goal_breakdown",
    "creative_idea_generation",
    "judgment_decision",
    "judgment_hr_decision",
    "idea_validation",
    "paraphrase",
    "candidate_screening"
]

ind = 0
for i in type:
    type2ind[i] = ind
    ind2type[ind] = i
    ind += 1


class InputData(BaseModel):
    text: str
    # Add other fields as needed


def predict_test(input_data: InputData):
    try:
        # Load the model from checkpoint
        checkpoint_path = '../model/ellaisthatyou.ckpt'
        model = BertClassifier.load_from_checkpoint(checkpoint_path)
        # Assuming your model and tokenizer are already defined as follows:
        model_name = "clicknext/phayathaibert"  # Change this to your model's name if necessary
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize device and model (your model already has this part)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        def tokenize_test_dataset(dataset):
            encoded = tokenizer(
                dataset['sentence'],
                # dataset['sentence'],
                padding='max_length',
                max_length=256,  # 128
                truncation=True,
            )
            return encoded

        # Assuming `test_df` is your test data frame containing 'sentence' column.
        test_df = pd.DataFrame({"sentence": [input_data.text]})
        test_dataset = Dataset.from_pandas(test_df)

        # Apply the tokenizer to the dataset
        test_dataset = test_dataset.map(tokenize_test_dataset, batched=True)

        # Set the format to be suitable for PyTorch (with input_ids and attention_mask)
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Prepare DataLoader for the test set
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Set the model to evaluation mode
        model.eval()

        predictions = []

        # Run inference
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass through the model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)  # Apply softmax for multi-class classification

                # Get the predicted class (index of highest probability)
                preds = torch.argmax(probs, dim=-1)
                print(preds)
                predictions.append(preds.cpu().numpy())  # Store predictions in CPU memory

        # Concatenate all predictions into a single array
        predicted_labels = np.concatenate(predictions, axis=0)
        print(probs.cpu().numpy())  # This prints the float scores for each class
        # Create a DataFrame to store predictions with a single column 'predicted_label'
        preds_df = pd.DataFrame(predicted_labels, columns=["predicted_label"])

        # Print out the predictions for the first 20 samples
        for text, label in zip(test_df["sentence"], predicted_labels):
            print(f"Text: {text}")
            print(f"Predicted Label: {ind2type[label]}")
            print("-" * 50)
            print("")

    except Exception as e:
        raise e

if __name__ == "__main__":
    predict_test(InputData(
        text="บอกข้าหน่อยโกโจชนะใช่ไหม?"))
    predict_test(InputData(
        text="มาเลยไอ้สัตว์ ควยเอ้ยยยยยยยย"))
    # uvicorn.run(app, host="0.0.0.0", port=8000)
