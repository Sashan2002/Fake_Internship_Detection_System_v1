"""
BERT sequence-classification wrapper for fraud detection (binary:
fraudulent vs not). See Section 8.1 and Experiment E5 (transformer-only,
Section 10 / Table "Model Training Experiments").
"""
from dataclasses import dataclass

from ml.nlp.tokenizer import TextTokenizer, TokenizerConfig

DEFAULT_CHECKPOINT = "bert-base-uncased"


@dataclass
class BertTrainingConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    random_seed: int = 42
    weight_decay: float = 0.01


class BertFraudClassifier:
    """
    Loads a HF AutoModelForSequenceClassification with num_labels=2.
    Training loop is left to a dedicated training script
    (see ml/models/transformer_model.py which selects between this and
    RoBERTa) to keep this class a reusable model definition rather than a
    script.
    """

    def __init__(self, config: BertTrainingConfig = None):
        self.config = config or BertTrainingConfig()
        self.tokenizer = TextTokenizer(TokenizerConfig(
            model_name=self.config.checkpoint, max_length=self.config.max_length
        ))
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.checkpoint, num_labels=2
            )
        return self._model

    def predict_proba(self, texts):
        """Returns fraud-class probability for each input text."""
        import torch

        model = self._load_model()
        model.eval()
        encodings = self.tokenizer.encode_batch(texts)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1].tolist()  # probability of class 1 = fraudulent

    def save(self, output_dir: str):
        model = self._load_model()
        model.save_pretrained(output_dir)
        self.tokenizer._load().save_pretrained(output_dir)

    @classmethod
    def load(cls, model_dir: str, config: BertTrainingConfig = None):
        instance = cls(config)
        from transformers import AutoModelForSequenceClassification
        instance._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.tokenizer = TextTokenizer(TokenizerConfig(model_name=model_dir))
        return instance
