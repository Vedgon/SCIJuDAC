"""
Custom collator to override the HF's data collator.
"""

from dataclasses import dataclass
import torch

@dataclass
class HierarchicalCollator:
	"""
	Written to replace the default collator used by HF. Supposed
	to work only for the SCIJuDAC setup.
	Pads tokenized input sequences to the maximum sequence length
	within each batch and collates the input IDs, attention masks,
	document-level labels, coarse labels, and fine-grained labels
	into tensors.
	"""
	pad_token_id: int = 0

	def __call__(self, features:list[dict[str, any]]) -> dict[str, torch.Tensor]:
		"""
		Collate tokenized features into a padded batch.

		Arguments
		---------
		features : list[dict[str, any]]
			Feature dictionaries produced during tokenization. Each
			dictionary contains `input_ids`, `attention_mask`,
			`labels`, `coarse_label`, and `fine_label`.

		Returns
		-------
		dict[str, torch.Tensor]
			A dictionary containing the padded `input_id`, 
			`attention_mask` tensors, along with `labels`,
			`coarse_label`, and `fine_label` tensors.
		"""
		batch_size = len(features)
		max_seq_len = max(f["input_ids"].shape[0] for f in features)

		input_ids = torch.full((batch_size, max_seq_len), fill_value=self.pad_token_id, dtype=torch.long)
		attention_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
		labels, coarse_labels, fine_labels = [], [], []

		for i, feature in enumerate(features):
			seq_len = feature["input_ids"].shape[0]
			input_ids[i, :seq_len] = feature["input_ids"]
			attention_masks[i, :seq_len] = feature["attention_mask"]

			labels.append(feature["labels"])
			coarse_labels.append(feature["coarse_label"])
			fine_labels.append(feature["fine_label"])

		labels = torch.tensor(labels)
		coarse_labels = torch.tensor(coarse_labels)
		fine_labels = torch.tensor(fine_labels)

		return {
			"input_ids": input_ids,
			"attention_mask": attention_masks,
			"labels": labels,
			"coarse_label": coarse_labels,
			"fine_label": fine_labels
		}
