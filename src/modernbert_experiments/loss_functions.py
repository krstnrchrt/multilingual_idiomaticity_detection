from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import torch
from transformers import PreTrainedModel, BatchEncoding
import torch.nn.functional as F


class LossFunction(Callable, ABC):

    @classmethod
    def init(
            cls,
            model: PreTrainedModel,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
    ):
        model_inputs = {
            k: v for k, v in inputs.items() if k in ["labels", "input_ids", "attention_mask"]
        }

        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        if logits.shape[-1] < 2:
            raise ValueError(
                "CrossEntropyLoss requires logits with at least 2 classes. Check your model configuration.")

        # Standard cross-entropy loss (make sure labels are correctly named and in range)
        labels = inputs.get("labels")
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return outputs, logits, loss

    @staticmethod
    @abstractmethod
    def get_identifier() -> str:
        raise NotImplementedError("Must return a unique identifier for this loss.")


    @abstractmethod
    def __call__(
            self,
            *args,
            model: PreTrainedModel,
            return_outputs: bool = False,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            **kwargs
    ):
        raise NotImplementedError("Must be implemented in loss!")


class VanillaCrossEntropyLoss(LossFunction):

    def __call__(
            self,
            *args,
            model: PreTrainedModel,
            return_outputs: bool = False,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            **kwargs
    ):
        outputs, logits, loss = self.init(model, inputs)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def get_identifier() -> str:
        return "vanilla_cross_entropy_loss"


class AbstractSimilarityLossFunction(LossFunction, ABC):

    def __init__(self, alpha: float):
        self.alpha: float = alpha

    @staticmethod
    def _ensure_compatible_model_outputs(outputs):
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("'hidden_states' must be defined in the models outputs!")

    @abstractmethod
    def __call__(
            self,
            *args,
            model: PreTrainedModel,
            return_outputs: bool = False,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            **kwargs
    ):
        raise NotImplementedError("Must be implemented in loss!")


class CosineSimilarityMWEWithTargetSentenceLoss(AbstractSimilarityLossFunction):

    def __call__(
            self,
            *args,
            model: PreTrainedModel,
            return_outputs: bool = False,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            **kwargs
    ):
        # Initialize: obtain model outputs, logits, and the initial loss
        outputs, logits, loss = self.init(model, inputs)
        labels = inputs.get("labels")
        labels_float = labels.float()

        self._ensure_compatible_model_outputs(outputs)

        # Get the last hidden states: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Retrieve custom index fields and squeeze any extra dimensions
        mwe_start = inputs.get("mwe_start_positions").squeeze().to(torch.long)
        mwe_end = inputs.get("mwe_end_positions").squeeze().to(torch.long)
        target_start = inputs.get("target_start_positions").squeeze().to(torch.long)
        target_end = inputs.get("target_end_positions").squeeze().to(torch.long)

        # Clamp the indices to the valid range [0, seq_len-1]
        mwe_start = mwe_start.clamp(min=0, max=seq_len - 1)
        mwe_end = mwe_end.clamp(min=0, max=seq_len - 1)
        # Ensure mwe_end is not less than mwe_start:
        mwe_end = torch.max(mwe_start, mwe_end)

        target_start = target_start.clamp(min=0, max=seq_len - 1)
        target_end = target_end.clamp(min=0, max=seq_len - 1)
        target_end = torch.max(target_start, target_end)

        # Create an index tensor for the sequence dimension
        # shape: [1, seq_len] to broadcast with each sample in the batch
        indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)  # [1, seq_len]

        # Build masks for the MWE region and the target region for each sample
        # mwe_mask: [batch_size, seq_len] where True indicates tokens in the MWE region
        mwe_mask = (indices >= mwe_start.unsqueeze(1)) & (indices <= mwe_end.unsqueeze(1))
        target_mask = (indices >= target_start.unsqueeze(1)) & (indices <= target_end.unsqueeze(1))

        # Define context as tokens in the target region excluding the MWE tokens
        context_mask = target_mask & (~mwe_mask)

        # Remove padding tokens using the attention mask (which is assumed to be [batch_size, seq_len])
        attention_mask = inputs.get("attention_mask").bool()
        context_mask = context_mask & attention_mask

        # Convert boolean masks to float (for weighted sum computation)
        mwe_mask_float = mwe_mask.float()  # [batch_size, seq_len]
        context_mask_float = context_mask.float()

        # Sum the token embeddings over the MWE region and average them
        # Using batch matrix multiplication to sum over the sequence dimension:
        # mwe_sum: [batch_size, hidden_dim]
        mwe_sum = torch.bmm(mwe_mask_float.unsqueeze(1), hidden_states).squeeze(1)
        mwe_counts = mwe_mask_float.sum(dim=1).unsqueeze(1).clamp(min=1)  # avoid division by zero
        mwe_embedding = mwe_sum / mwe_counts  # [batch_size, hidden_dim]

        # Similarly for the context (target sentence excluding MWE tokens)
        context_sum = torch.bmm(context_mask_float.unsqueeze(1), hidden_states).squeeze(1)
        context_counts = context_mask_float.sum(dim=1).unsqueeze(1).clamp(min=1)
        context_embedding = context_sum / context_counts  # [batch_size, hidden_dim]

        # Compute cosine similarity for each sample in the batch
        similarity_scores = F.cosine_similarity(mwe_embedding, context_embedding, dim=1)  # [batch_size]

        # Compute the auxiliary similarity loss: encourage high similarity for literal (label=1) and low for idiomatic (label=0)
        similarity_loss = -torch.mean(
            labels_float * similarity_scores + (1 - labels_float) * (1 - similarity_scores)
        )

        final_loss = loss + self.alpha * similarity_loss

        return (final_loss, outputs) if return_outputs else final_loss

    @staticmethod
    def get_identifier() -> str:
        return "custom_loss_1"


class CosineSimilarityTargetWithContextSentencesLoss(AbstractSimilarityLossFunction):

    def __call__(
            self,
            *args,
            model: PreTrainedModel,
            return_outputs: bool = False,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            **kwargs
    ):
        # Initialize: obtain model outputs, logits, and the initial loss
        outputs, logits, loss = self.init(model, inputs)
        labels = inputs.get("labels")
        labels_float = labels.float()

        self._ensure_compatible_model_outputs(outputs)

        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Retrieve custom index fields and squeeze any extra dimensions
        target_start = inputs.get("target_start_positions").squeeze().to(torch.long)
        target_end = inputs.get("target_end_positions").squeeze().to(torch.long)

        # Clamp the indices to the valid range [0, seq_len-1]
        target_start = target_start.clamp(min=0, max=seq_len - 1)
        target_end = target_end.clamp(min=0, max=seq_len - 1)
        target_end = torch.max(target_start, target_end)

        # Create an index tensor for the sequence dimension
        # shape: [1, seq_len] to broadcast with each sample in the batch
        indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # Build masks for the target region and the sequence for each sample
        target_mask = (indices >= target_start.unsqueeze(1)) & (indices <= target_end.unsqueeze(1))
        sequence_mask = (indices <= seq_len)

        # Define context mask excluding the target sentence
        context_mask = sequence_mask & (~target_mask)

        # Remove padding tokens using the attention mask (which is assumed to be [batch_size, seq_len])
        attention_mask = inputs.get("attention_mask").bool()
        context_mask = context_mask & attention_mask

        # Convert boolean mask to float (for weighted sum computation)
        target_mask_float = target_mask.float()
        context_mask_float = context_mask.float()

        target_sum = torch.bmm(target_mask_float.unsqueeze(1), hidden_states).squeeze(1)
        target_counts = target_mask_float.sum(dim=1).unsqueeze(1).clamp(min=1)  # avoid division by zero
        target_embedding = target_sum / target_counts  # [batch_size, hidden_dim]

        # Similarly for the context (whole sequence excluding target sentence tokens)
        context_sum = torch.bmm(context_mask_float.unsqueeze(1), hidden_states).squeeze(1)
        context_counts = context_mask_float.sum(dim=1).unsqueeze(1).clamp(min=1)
        context_embedding = context_sum / context_counts

        # Compute cosine similarity for each sample in the batch
        similarity_scores = F.cosine_similarity(target_embedding, context_embedding, dim=1)

        # Compute the auxiliary similarity loss: encourage high similarity for literal (label=1) and low for idiomatic (label=0)
        similarity_loss = -torch.mean(
            labels_float * similarity_scores + (1 - labels_float) * (1 - similarity_scores)
        )

        final_loss = loss + self.alpha * similarity_loss

        return (final_loss, outputs) if return_outputs else final_loss

    @staticmethod
    def get_identifier() -> str:
        return "custom_loss_2"


class TargetSentenceSpecificTokenEmbeddingsBasedEntropyLoss(AbstractSimilarityLossFunction):

    def __call__(
            self,
            *args,
            model: PreTrainedModel,
            return_outputs: bool = False,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            **kwargs
    ):
        # Initialize: obtain model outputs, logits, and the initial loss
        outputs, logits, loss = self.init(model, inputs)
        labels = inputs.get("labels")
        labels_float = labels.float()

        self._ensure_compatible_model_outputs(outputs)

        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Determine entropy in target sentence a
        pass

    @staticmethod
    def get_identifier() -> str:
        return "custom_loss_3"
