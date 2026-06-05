"""
Computes different evaluation metrics from the logits.
The final prediction for a file is obtained by applying
`argmax` to the averaged logits.
"""
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

from datasets import Dataset
from transformers import EvalPrediction

class ComputeMetrics():
	"""
	Computes classification metrics for model evaluation.

	Supports both token-level and file-level evaluation.
	The metrics are computed from the model predictions.

	Metrics computed:
	- Accuracy
	- Macro Recall
	- Weighted Recall
	- Macro F1
	- Weighted F1
	"""
	def __init__(self, data:Dataset):
		"""
		Initializes the metric computation object.

		Parameters
		----------
		data : Dataset
			Test dataset containing file identifiers.
		"""
		self.file_ids = data["file_id"]


	def __call__(self, eval_pred:EvalPrediction|tuple) -> dict[str, float]:
		"""
		Computes evaluation metrics from model predictions.

		Arguments
		---------
		eval_pred : EvalPrediction | tuple
			Evaluation output produced by the trainer. Can be either an
			EvalPrediction instance or a tuple containing logits and labels.

		Returns
		-------
		dict[str, float]
			Dictionary containing the following metrics:
			- accuracy
			- recall_macro
			- recall_weighted
			- f1_macro
			- f1_weighted
		"""
		# Condition to check whether `eval_pred` is a `EvalPrediction` instance or a tuple.
		# Extracts the logits and labels accordingly.
		if hasattr(eval_pred, "predictions"): logits, labels = eval_pred.predictions, eval_pred.label_ids
		else: logits, labels = eval_pred

		final_preds = np.argmax(logits, axis=1)
		final_labels = labels

		return {
			"accuracy": accuracy_score(final_labels, final_preds),
			"recall_macro": recall_score(final_labels, final_preds, average="macro", zero_division=0),
			"recall_weighted": recall_score(final_labels, final_preds, average="weighted", zero_division=0),
			"f1_macro": f1_score(final_labels, final_preds, average="macro"),
			"f1_weighted": f1_score(final_labels, final_preds, average="weighted", zero_division=0),
		}
