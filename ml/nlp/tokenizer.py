"""
Tokenizer wrapper used identically for BERT and RoBERTa representations.

Per Section 8.1, the tokenizer, transformer version, maximum sequence
length, hyperparameters, random seed and dataset version must all be
recorded for reproducibility (Section 17). This wrapper stores those
alongside itself so experiment logs can serialise `TokenizerConfig` as-is.
"""
from dataclasses import dataclass, asdict


@dataclass
class TokenizerConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True

    def to_dict(self):
        return asdict(self)


class TextTokenizer:
    """
    Thin wrapper around a Hugging Face AutoTokenizer. Import of
    `transformers` is deferred to first use so that modules which don't
    need the transformer stack (e.g. the TF-IDF baseline) can be imported
    without requiring torch/transformers to be installed.
    """

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    def encode_batch(self, texts):
        tokenizer = self._load()
        return tokenizer(
            list(texts),
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def encode_single(self, text: str):
        return self.encode_batch([text])
