from typing import List

import torch
import torch.nn.functional as F

# change it with respect to the original model
from .config import LlamaConfig
from .llama import load_pretrained
from .tokenizer import Tokenizer


class LlamaZeroShotClassifier(torch.nn.Module):
    """Zero-shot classifier using Llama model."""

    def __init__(
        self, config: LlamaConfig, tokenizer: Tokenizer, label_names: List[str]
    ):
        """Initialize the LlamaZeroShotClassifier.

        Args:
            config (LlamaConfig): Configuration object for the Llama model.
            tokenizer (Tokenizer): Tokenizer object for the Llama model.
            label_names (List[str]): List of label names.

        Attributes:
            num_labels (int): Number of labels in the classification task.
            llama (Llama): Llama model.
            tokenizer (Tokenizer): Tokenizer object for the Llama model.
            label_name_ids (List[List[int]]): List of label token IDs.
        """

        super(LlamaZeroShotClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)

        # Zero-shot classification does not require updating llama paramters.
        for param in self.llama.parameters():
            param.requires_grad = False
        assert len(label_names) == self.num_labels

        self.tokenizer = tokenizer
        self.label_name_ids = [
            tokenizer.encode(label, bos=False, eos=False) for label in label_names
        ]

    def forward(self, input_ids):
        # compute the completion probability of each label string
        logits, _ = self.llama(input_ids)
        log_probabilities = F.log_softmax(logits, dim=-1)
        label_probabilities = torch.zeros(
            (log_probabilities.shape[0], self.num_labels),
            device=log_probabilities.device,
        )
        for i, label_token_ids in enumerate(self.label_name_ids):
            total_log_prob = torch.sum(
                log_probabilities[:, :, label_token_ids], axis=-1
            )
            label_probabilities[:, i] = total_log_prob[:, 0]
        return label_probabilities


class LlamaEmbeddingClassifier(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        """Initialize the LlamaEmbeddingClassifier.

        Args:
            config (LlamaConfig): Configuration object for the Llama model.

        Attributes:
            num_labels (int): Number of labels in the classification task.
            llama (Llama): Llama model.
            dropout (torch.nn.Dropout): Dropout layer.
            classifier_head (torch.nn.Linear): Classifier head.
        """
        super(LlamaEmbeddingClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)

        # If we use pretrain mode, we freeze Llama parameters.
        for param in self.llama.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass of the LlamaEmbeddingClassifier.

        Args:
            input_ids (torch.Tensor): Input token IDs.

        Returns:
            torch.Tensor: Log-probabilities over all classes.
        """
        # Returns a tuple of (logits, hidden_states)
        _, hidden_states = self.llama(input_ids)
        # Take the hidden state of the last token of the input sequence
        last_hidden_state = hidden_states[:, -1, :]

        # Apply dropout to the hidden state at training time to mitigate overfitting.
        pooled_output = self.dropout(last_hidden_state)

        # Get logits (unnormalized probabilities) over all classes.
        logits = self.classifier_head(pooled_output)

        # Get log-probabilities over all classes.
        return F.log_softmax(logits, dim=-1)
