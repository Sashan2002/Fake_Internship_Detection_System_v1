"""
RoBERTa sequence-classification wrapper, structurally identical to
BertFraudClassifier so the two can be compared fairly as planned in
Section 10 ('compare BERT/RoBERTa as planned' - dev sequence step 9).
"""
from dataclasses import dataclass

from ml.nlp.tokenizer import TextTokenizer, TokenizerConfig
from ml.nlp.bert_model import BertFraudClassifier, BertTrainingConfig

DEFAULT_CHECKPOINT = "roberta-base"


@dataclass
class RobertaTrainingConfig(BertTrainingConfig):
    checkpoint: str = DEFAULT_CHECKPOINT


class RobertaFraudClassifier(BertFraudClassifier):
    """
    Reuses BertFraudClassifier's logic entirely (AutoModel/AutoTokenizer
    make the HF API checkpoint-agnostic); this subclass exists so
    experiment configs and saved artefacts stay clearly labelled by
    architecture for the required BERT-vs-RoBERTa comparison.
    """

    def __init__(self, config: RobertaTrainingConfig = None):
        super().__init__(config or RobertaTrainingConfig())
