import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformers import (
	AutoTokenizer,
	DataCollatorWithPadding
)
from datasets import Dataset
from tqdm import tqdm

from config.config import CHECKPOINTS_ROOT
from training import (
	load_and_tokenize_data,
	get_checkpoint_path,
	BicephalousConfig,
	BicephalousClassifier
)


class DataAndModelLoader():
	def __init__(
				self, model_name:str, dataset_path:Path,
				num_labels:int, batch_size:int, seed:int,
				load_lc_ckpt:str, load_cc_ckpt:str,
				load_full_ckpt:str, enable_quantization:str
	) -> None:
		"""
		Loads model checkpoints and datasets and extracts model outputs
		required for downstream analysis.

		The loader supports both flat and hierarchical classifiers and
		handles label-collapsed, classifier-collapsed, and full-model
		checkpoints.

		Parameters
		----------
		model_name : str
			Name of the pretrained backbone model.

		dataset_path : Path
			Path to the dataset file.

		num_labels : int
			Number of target classes.

		batch_size : int
			Batch size used during feature extraction.

		seed : int
			Random seed used for reproducibility.

		load_lc_ckpt : str
			Flag indicating whether to load a label-collapsed checkpoint.

		load_cc_ckpt : str
			Flag indicating whether to load a classifier-collapsed
			checkpoint.

		load_full_ckpt : str
			Flag indicating whether to load the full-model checkpoint.

		enable_quantization : str
			Flag indicating whether model quantization is enabled.
		"""
		self.model_name = model_name
		self.dataset_path = dataset_path
		self.num_labels = num_labels
		self.batch_size = batch_size
		self.seed = seed
		self.collapse_labels = load_lc_ckpt
		self.collapse_classifier = load_cc_ckpt
		self.load_full_ckpt = load_full_ckpt
		self.enable_quantization = enable_quantization


	@staticmethod
	def detect_ckpt_format(ckpt_path:Path):
		"""
		Detect the format of the model checkpoint.

		Arguments
		---------
		ckpt_path : Path
			Path of the checkpoint to be loaded.

		Returns
		-------
		str | None
			Checkpoint format. Returns "sharded" for a sharded
			Safetensors checkpoint, "safetensor" for a single
			Safetensors checkpoint, and "bin" for a PyTorch
			binary checkpoint. Returns `None` if no supported
			checkpoint format is found.
		"""
		formats = {
			"model.safetensors.index.json": "sharded",
			"model.safetensors": "safetensor",
			"pytorch_model": "bin"
		}

		for file, format in formats.items():
			if (ckpt_path / file).exists():
				return format


	def load_model(self):
		"""
		Load the model and tokenizer from the selected checkpoint.

		The checkpoint is selected according to the dataset, model,
		and checkpoint configuration specified when initializing
		the loader. The model weights are loaded from either a
		Safetensors or PyTorch binary checkpoint.

		Returns
		-------
		model : BicephalousClassifier
			Classifier initialized with the checkpoint weights and
			moved to the available compute device.

		tokenizer : AutoTokenizer
			Tokenizer corresponding to the loaded model checkpoint.
		"""
		model_ckpt_path = get_checkpoint_path(
			CHECKPOINTS_ROOT, self.dataset_path,
			self.model_name, self.collapse_labels,
			self.collapse_classifier, self.load_full_ckpt
		)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
		config = BicephalousConfig.from_pretrained(model_ckpt_path)
		backbone = BicephalousClassifier.load_backbone(config, tokenizer)
		model = BicephalousClassifier(config=config, backbone=backbone)

		state_dict_format = self.detect_ckpt_format(model_ckpt_path)
		if state_dict_format == "sharded":
			from safetensors.torch import load_file
			state_dict = {}
			for shard_path in sorted(model_ckpt_path.glob("*.safetensors")):
				state_dict.update(load_file(str(shard_path)))
			backbone.load_state_dict(state_dict, strict=False)

		elif state_dict_format == "safetensor":
			from safetensors.torch import load_file
			state_dict = load_file(str(model_ckpt_path / "model.safetensors"))

		elif state_dict_format == "bin":
			state_dict = torch.load(model_ckpt_path / "pytorch_model.bin", map_location="cpu")

		model.load_state_dict(state_dict, strict=False)
		model.to(device)

		return model, tokenizer


	def load_dataset(self, tokenizer:AutoTokenizer):
		"""
		Load and tokenize the test split of the dataset.

		Parameters
		----------
		tokenizer : AutoTokenizer
			Tokenizer corresponding to the loaded model.

		Returns
		-------
		test_data : Dataset
			Tokenized test split of the selected dataset.
		"""
		_, _, test_data = load_and_tokenize_data(self, tokenizer)

		return test_data


	def extract_features(self, model:BicephalousClassifier, tokenizer:AutoTokenizer, data:Dataset):
		"""
		Extract logits, embeddings, and labels from the input data.

		The model is evaluated without gradient computation. For a
		classifier-collapsed model, logits from the flat classification
		head are collected. Otherwise, logits from the coarse and fine
		classification heads are collected. Last-layer embeddings are
		L2-normalized before being returned.

		Parameters
		----------
		model : BicephalousClassifier
			Trained classifier used for feature extraction.

		tokenizer : AutoTokenizer
			Tokenizer corresponding to the loaded model.

		data : Dataset
			Tokenized dataset from which features are extracted.

		Returns
		-------
		logits : Tensor | list
			Logits produced by the flat classification head when the
			classifier-collapsed model is used. Otherwise, an empty
			list.

		coarse_logits : Tensor | list
			Logits produced by the coarse classification head for a
			hierarchical model. Otherwise, an empty list.

		fine_logits : Tensor | None | list
			Logits produced by the fine classification head. ``None``
			if the fine head produces no logits, or an empty list when
			using the classifier-collapsed model.

		embeddings : Tensor
			L2-normalized last-layer embeddings for all input samples.

		labels : Tensor
			Ground-truth class labels corresponding to the extracted
			features.
		"""
		model.eval()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)

		collator = DataCollatorWithPadding(tokenizer=tokenizer)
		dataloader = DataLoader(data.with_format(None), batch_size=self.batch_size, collate_fn=collator)

		logits, coarse_logits, fine_logits, embeddings, labels, = [], [], [], [], []

		with torch.no_grad():
			for batch in tqdm(dataloader, desc="Feature extraction"):
				inputs = {
					"input_ids": batch["input_ids"].to(device),
					"attention_mask": batch["attention_mask"].to(device)
				}

				outputs = model(**inputs, output_hidden_states=True)

				if self.collapse_classifier == "y":
					logits.append(outputs["logits"].detach().cpu())
				
				else:
					coarse_logits.append(outputs["coarse_logits"].detach().cpu())
					if outputs["fine_logits"] is not None:
						fine_logits.append(outputs["fine_logits"].detach().cpu())

				emb = outputs["last_layer_emb"].detach().to(torch.float64)
				norm = emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
				emb = (emb / norm).cpu()
				embeddings.append(emb)

				labels.append(batch["labels"].cpu())

		if self.collapse_classifier == "y":
			logits = torch.cat(logits, dim=0)
		else:
			coarse_logits = torch.cat(coarse_logits, dim=0)
			fine_logits = (
				torch.cat(fine_logits, dim=0)
				if len(fine_logits) > 0
				else None
			)
		embeddings = torch.cat(embeddings, dim=0)
		labels = torch.cat(labels, dim=0)

		return logits, coarse_logits, fine_logits, embeddings, labels


	def get_preds(self, logits:Tensor, coarse_logits:Tensor, fine_logits:Tensor, labels:Tensor):
		"""
		Compute class probabilities and predictions from model logits.

		For a classifier-collapsed model, probabilities are computed
		directly from the flat classifier logits. For a hierarchical
		model, coarse and fine probabilities are combined to obtain
		probabilities over the final classes.

		Parameters
		----------
		logits : Tensor | list
			Logits from the flat classification head. Used when the
			classifier-collapsed model is loaded.

		coarse_logits : Tensor | list
			Logits from the coarse classification head.

		fine_logits : Tensor | None | list
			Logits from the fine classification head. May be ``None``
			when the fine head is unavailable.

		labels : Tensor
			Ground-truth class labels.

		Returns
		-------
		final_probs : Tensor
			Class probabilities for each input sample.

		final_preds : Tensor
			Predicted class indices obtained by selecting the class
			with the highest probability.

		final_labels : Tensor
			Ground-truth class labels corresponding to the predictions.
		"""
		if self.collapse_classifier == "y":
			final_probs = torch.softmax(logits, dim=-1)

		else:
			coarse_probs = torch.softmax(coarse_logits, dim=-1)

			if fine_logits is not None:
				fine_probs = torch.softmax(fine_logits, dim=-1)
				final_probs = torch.zeros(coarse_logits.size(0), 3)

				final_probs[:, 0] = coarse_probs[:,0] 				    # dismissed
				final_probs[:, 1] = coarse_probs[:,1] * fine_probs[:,1] # allowed
				final_probs[:, 2] = coarse_probs[:,1] * fine_probs[:,0] # partly allowed

			else:
				final_probs = coarse_probs

		final_preds = torch.argmax(final_probs, axis=-1)
		final_labels = labels

		final_labels = torch.tensor(final_labels) if not isinstance(final_labels, torch.Tensor) else final_labels

		return final_probs, final_preds, final_labels
