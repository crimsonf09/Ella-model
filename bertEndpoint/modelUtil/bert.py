import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score

class BertClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int = 9, learning_rate: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Accumulators
        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        self.val_preds.extend(preds.cpu().tolist())
        self.val_labels.extend(batch["labels"].cpu().tolist())
        self.log("val_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def on_validation_epoch_end(self):
        f1 = f1_score(self.val_labels, self.val_preds, average="micro")
        self.log("val_f1", f1, prog_bar=True)
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        self.test_preds.extend(preds.cpu().tolist())
        self.test_labels.extend(batch["labels"].cpu().tolist())
        return outputs.loss

    def on_test_epoch_end(self):
        f1 = f1_score(self.test_labels, self.test_preds, average="micro")
        self.log("test_f1", f1)
        print(f"\n🔍 Test F1 Score (micro): {f1:.4f}")
        self.test_preds.clear()
        self.test_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)