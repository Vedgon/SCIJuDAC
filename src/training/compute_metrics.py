from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class ComputeMetrics():
	def __init__(self, data, all_toks):
		self.file_ids = data["file_id"] if all_toks else None
		self.all_toks = all_toks
		# print(f"Data loaded in `ComputeMetrics`:\n{data}")
		# print(f"File IDs in eval dataset: {set(data["file_id"])}")
		# print(f"File IDs stored in the local variable: {self.file_ids}")


	def __call__(self, eval_pred):
		# print(eval_pred)
		if hasattr(eval_pred, "predictions"): logits, labels = eval_pred.predictions, eval_pred.label_ids
		else: logits, labels = eval_pred
		# print(f"{logits},\n\n{labels}")
		# print(f"No: of logits: {len(logits)}, No. of labels: {len(labels)}")
		# print(f"No. of chunks: {len(self.file_ids)}")
		# print(f"File(s): {set(self.file_ids)}")

		if self.all_toks:
			per_file_logits = defaultdict(list)
			per_file_labels = {}

			for fID, logit, label in zip(self.file_ids, logits, labels):
				per_file_logits[fID].append(logit)
				per_file_labels[fID] = label

			final_preds, final_labels = [], []
			for fID in per_file_logits:
				# print(f"{fID}: {len(per_file_logits[fID])}")
				aggregated_logits = np.mean(per_file_logits[fID], axis=0)
				final_preds.append(np.argmax(aggregated_logits))
				final_labels.append(per_file_labels[fID])

		else:
			final_preds = np.argmax(logits, axis=1)
			final_labels = labels

		# print(f"Predictions: {final_preds}, Actual: {final_labels}")

		return {
			"f1_macro": f1_score(final_labels, final_preds, average="macro"),
			"f1_weighted": f1_score(final_labels, final_preds, average="weighted"),
			"accuracy": accuracy_score(final_labels, final_preds)
		}
