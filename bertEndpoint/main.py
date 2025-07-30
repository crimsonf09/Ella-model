from contextlib import asynccontextmanager
import pandas as pd
import uvicorn
from fastapi import Request, HTTPException
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modelUtil.bert import BertClassifier
from datasets import Dataset
import numpy as np
from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.responses import JSONResponse

model = None
tokenizer = None
device = None
type2ind = {}
ind2type = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device, type2ind, ind2type

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

    checkpoint_path = 'model/ellaisthatyou.ckpt'
    model_name = "clicknext/phayathaibert"

    model = BertClassifier.load_from_checkpoint(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    yield

    print("shutting down")


app = FastAPI(lifespan=lifespan)


class InputData(BaseModel):
    text: str



def load_model(checkpoint_path):
    try:
        raise Exception
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
async def ping():
    try:
        return {"message": "Hello, World!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/label")
async def label():
    try:
        return {
        '1': 'research_insight',
        '2' : 'strategy_planning',
        '3' : 'goal_breakdown',
        '4' : 'creative_idea_generation',
        '5' : 'judgment_decision',
        '6' : 'judgment_hr_decision',
        '7' : 'idea_validation',
        '8' : 'paraphrase',
        '9' : 'candidate_screening'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    return {"error": str(exc)}


@app.post("/predict")
async def predict(input_data: InputData = Body(...)):
    try:
        return await make_prediction(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def make_prediction(input_data: InputData):
    global model, tokenizer, device

    # model_name = "clicknext/phayathaibert"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    #print(tokenizer)

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
    predictions = []

    # Run inference
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
            #print(preds)
            predictions.append(preds.cpu().numpy())  # Store predictions in CPU memory

    # Concatenate all predictions into a single array
    predicted_labels = np.concatenate(predictions, axis=0)
    # Create a DataFrame to store predictions with a single column 'predicted_label'
    preds_df = pd.DataFrame(predicted_labels, columns=["predicted_label"])

    # Print out the predictions for the first 20 samples
    output = {
        "input_text" : "",
        "predicted_label" :"",
        "predicted_array" : probs.cpu().numpy()[0].tolist()
    }

    for text, label in zip(test_df["sentence"], predicted_labels):
        output["input_text"] = text
        output["predicted_label"] = ind2type[label]
        print("oka....")

    return JSONResponse(output)


if __name__ == "__main__":
    # predict_test(InputData(text="teach me lighting Pytorch. Bert, please."))
    uvicorn.run(app, host="0.0.0.0", port=8080)
