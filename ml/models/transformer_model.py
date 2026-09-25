"""
Transformer-only model selection wrapper (Experiment E5, Section 10).

Chooses between BertFraudClassifier and RobertaFraudClassifier under a
single interface so ml/evaluation/evaluate.py and
ml/models/integrated_model.py don't need to know which architecture is
active - required for the planned controlled BERT-vs-RoBERTa comparison
(dev sequence step 9) to be a fair, swappable comparison.
"""
from dataclasses import dataclass

from ml.nlp.bert_model import BertFraudClassifier, BertTrainingConfig
from ml.nlp.roberta_model import RobertaFraudClassifier, RobertaTrainingConfig

ARCHITECTURES = {
    "bert": (BertFraudClassifier, BertTrainingConfig),
    "roberta": (RobertaFraudClassifier, RobertaTrainingConfig),
}


@dataclass
class TransformerModelConfig:
    architecture: str = "bert"  # "bert" | "roberta"


def build_transformer_model(config: TransformerModelConfig = None, training_config=None):
    config = config or TransformerModelConfig()
    if config.architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{config.architecture}'. Choose from {list(ARCHITECTURES)}")
    model_cls, config_cls = ARCHITECTURES[config.architecture]
    return model_cls(training_config or config_cls())
