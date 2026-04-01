import json
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    fbeta_score,
    precision_recall_curve,
    auc,
)
from transformers import (
    PreTrainedModel,
    BatchEncoding,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer import Trainer
import torch

import src.settings as sts
from src.modernbert_experiments.loss_functions import LossFunction


class SlidTrainer(Trainer):
    def __init__(self, *args, loss_function: LossFunction | None = None, **kwargs):
        super().__init__(
            *args,
            compute_metrics=self.compute_metrics,
            **kwargs
        )
        self.loss_function: LossFunction = loss_function

    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Union[dict, BatchEncoding],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            # If the model returns hidden_states, only keep the last one.
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # Replace hidden_states with a tuple containing only the last hidden state.
                outputs.hidden_states = (outputs.hidden_states[-1],)
            # Get loss, logits and labels from the output.
            # Note: Depending on your model, loss might be None during prediction.
            loss = outputs.loss if hasattr(outputs, "loss") else None
            # Use outputs.logits if available, otherwise assume outputs is a tuple.
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            labels = inputs.get("labels")

        if prediction_loss_only:
            return loss, None, None

        return loss, logits, labels

    def compute_loss(
            self,
            model: PreTrainedModel,
            inputs: Union[dict[str, torch.Tensor], BatchEncoding],
            return_outputs: bool = False,
            num_items_in_batch: int = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        if self.loss_function is None:
            return super().compute_loss(
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch
            )
        return self.loss_function(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch
        )

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        if logits.ndim == 1 or logits.shape[-1] == 1:
            predictions = (logits >= 0).astype(int)
        else:
            predictions = np.argmax(logits, axis=1)

        # Accuracy
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        accuracy_balanced = balanced_accuracy_score(y_true=labels, y_pred=predictions)

        # F1
        f1_binary = f1_score(y_true=labels, y_pred=predictions, pos_label=sts.CLASS_POSITIVE, average="binary", zero_division=0)
        f1_micro = f1_score(y_true=labels, y_pred=predictions, average="micro", zero_division=0)
        f1_macro = f1_score(y_true=labels, y_pred=predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true=labels, y_pred=predictions, average="weighted", zero_division=0)

        # F2
        f2_binary = fbeta_score(y_true=labels, y_pred=predictions, pos_label=sts.CLASS_POSITIVE, average="binary", beta=2, zero_division=0)
        f2_micro = fbeta_score(y_true=labels, y_pred=predictions, average="micro", beta=2, zero_division=0)
        f2_macro = fbeta_score(y_true=labels, y_pred=predictions, average="macro", beta=2, zero_division=0)
        f2_weighted = fbeta_score(y_true=labels, y_pred=predictions, average="weighted", beta=2, zero_division=0)

        # Precision
        precision_binary = precision_score(y_true=labels, y_pred=predictions, pos_label=sts.CLASS_POSITIVE, average="binary", zero_division=0)
        precision_micro = precision_score(y_true=labels, y_pred=predictions, average="micro", zero_division=0)
        precision_macro = precision_score(y_true=labels, y_pred=predictions, average="macro", zero_division=0)
        precision_weighted = precision_score(y_true=labels, y_pred=predictions, average="weighted", zero_division=0)

        # Recall
        recall_binary = recall_score(y_true=labels, y_pred=predictions, pos_label=sts.CLASS_POSITIVE, average="binary", zero_division=0)
        recall_micro = recall_score(y_true=labels, y_pred=predictions, average="micro", zero_division=0)
        recall_macro = recall_score(y_true=labels, y_pred=predictions, average="macro", zero_division=0)
        recall_weighted = recall_score(y_true=labels, y_pred=predictions, average="weighted", zero_division=0)

        # ROC AUC
        try:
            roc_auc = roc_auc_score(labels, logits[:, 1])
        except (Exception,):
            roc_auc = float("nan")

        try:
            precision_curve, recall_curve, _ = precision_recall_curve(labels, logits[:, 1])
            auprc = auc(recall_curve, precision_curve)
        except (Exception,):
            auprc = float("nan")

        # Confusion Matrix
        cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # TPR, TNR, FNR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / Recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true=labels, y_pred=predictions)

        return {
            "accuracy": accuracy,
            "accuracy_balanced": accuracy_balanced,

            "f1": f1_binary,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,

            "f2": f2_binary,
            "f2_micro": f2_micro,
            "f2_macro": f2_macro,
            "f2_weighted": f2_weighted,

            "precision": precision_binary,
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,

            "recall": recall_binary,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,

            "roc_auc": roc_auc,
            "auprc": auprc,
            "mcc": mcc,

            "tpr": tpr,
            "tnr": tnr,
            "fnr": fnr,

            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }


class LogCallback(TrainerCallback):
    def __init__(self, dir_artifacts: Path):
        self.dir_artifacts: Path = dir_artifacts
        self.metrics_time_series: dict = {}

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {})
        training_loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        learning_rate = logs.get("learning_rate")

        if training_loss is not None:
            if "train_loss" not in self.metrics_time_series:
                self.metrics_time_series["train_loss"] = []
            if "grad_norm" not in self.metrics_time_series:
                self.metrics_time_series["grad_norm"] = []
            if "learning_rate" not in self.metrics_time_series:
                self.metrics_time_series["learning_rate"] = []

            self.metrics_time_series["train_loss"].append(training_loss)
            self.metrics_time_series["grad_norm"].append(grad_norm)
            self.metrics_time_series["learning_rate"].append(learning_rate)

        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", {})
        for key, value in metrics.items():
            if key not in self.metrics_time_series:
                self.metrics_time_series[key] = []
            self.metrics_time_series[key].append(value)
        return control


    def on_train_end(self, args, state, control, **kwargs):
        record = {
            "run_name": args.run_name,
            "config": args.to_dict(),
            "metrics": self.metrics_time_series
        }
        with open(self.dir_artifacts / sts.FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_SUMMARY, "w") as f:
            json.dump(record, f, indent=2)
        return control
