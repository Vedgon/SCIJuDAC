import torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from transformers import (
	Trainer,
	PretrainedConfig,
	PreTrainedModel,
	AutoModel,
	AutoTokenizer
)
from peft import prepare_model_for_kbit_training, get_peft_model

from .config_quantization import quantization_config

from torch import Tensor
from transformers import PreTrainedTokenizerBase
from pathlib import Path

IGNORE_INDEX = -100 # Equals to the value set during dataset tokenization

class ComputeWeightsAndLosses():
	"""
	Utility class for computing class weights and training losses.

	Provides functionality for deriving class weights from label
	distributions and computing the combined coarse- and fine-grained
	classification loss used during training.
	"""
	@staticmethod
	def compute_weights(
		labels:Tensor, num_classes:int=2, fine_flag:bool=False
	)-> Tensor:
		"""
		Computes normalized class weights from a label distribution.

		For fine-grained classification, labels equal to `IGNORE_INDEX`
		are excluded from the weight computation.

		Arguments
		---------
		labels : Tensor
			Collection of class labels.

		num_classes : int, default=2
			Total number of classes.

		fine_flag : bool, default=False
			If True, excludes labels equal to `IGNORE_INDEX` before
			computing class frequencies.

		Returns
		-------
		weights : Tensor
			Tensor of shape `(num_classes,)` containing normalized class
			weights. If no valid labels are present, uniform weights are
			returned.
		"""
		if fine_flag:
			valid_labels = [l for l in labels if l != IGNORE_INDEX]
			counts = Counter(valid_labels)
		else: counts = Counter(labels)
		total = sum(counts.values())

		weights = []
		for i in range(num_classes):
			if counts[i] > 0: weights.append(total / counts[i])
			else: weights.append(0.0)

		weights = torch.tensor(weights, dtype=torch.float)

		if weights.sum() > 0:
			weights = weights / weights.sum()
		else: weights = torch.ones(num_classes, dtype=torch.float) / num_classes

		return weights


	@staticmethod
	def compute_loss(
		outputs:dict[str, Tensor], coarse_label:Tensor, fine_label:Tensor,
		lam:float, coarse_weights=Tensor|None, fine_weights=Tensor|None
	) -> tuple[Tensor, Tensor, Tensor]:
		"""
		Computes the hierarchical training loss.

		The total loss is defined as the sum of the coarse classification
		loss and a weighted fine classification loss:

		total_loss = coarse_loss + lam * fine_loss

		Fine-grained loss is computed only for examples whose fine label is
		not equal to `IGNORE_INDEX`.

		Arguments
		---------
		outputs : dict[str, Tensor]
			Model outputs containing the keys:
			- ``coarse_logits`` : shape `(batch_size, 2)`
			- ``fine_logits`` : shape `(batch_size, 2)` or ``None``

		coarse_label : Tensor
			Coarse-grained target labels.
			Shape: `(batch_size,)`

		fine_label : Tensor
			Fine-grained target labels.
			Shape: `(batch_size,)`
			Positions equal to `IGNORE_INDEX` are ignored when computing
			the fine-grained loss.

		lam : float
			Weight assigned to the fine-grained loss component.

		coarse_weights : Tensor | None, optional
			Class weights used for coarse-grained classification.

		fine_weights : Tensor | None, optional
			Class weights used for fine-grained classification.

		Returns
		-------
		total_loss : Tensor
			Total cross-entropy loss.

		coarse_loss : Tensor
			Coarse head cross-entropy loss.

		fine_loss : Tensor
			Fine head cross-entropy loss.
		"""
		coarse_logits = outputs["coarse_logits"]
		coarse_loss = F.cross_entropy(
			coarse_logits,
			coarse_label,
			weight=coarse_weights
		)

		fine_logits = outputs["fine_logits"]
		if fine_logits is not None:
			valid_fine_mask = fine_label != IGNORE_INDEX

			if valid_fine_mask.any():
				fine_loss = F.cross_entropy(fine_logits[valid_fine_mask], fine_label[valid_fine_mask], weight=fine_weights)
			else:
				fine_loss = torch.tensor(0.0, device=coarse_logits.device)

		else:
			fine_loss = torch.tensor(0.0, device=coarse_logits.device)

		total_loss = coarse_loss + lam * fine_loss

		return total_loss, coarse_loss, fine_loss


class BicephalousConfig(PretrainedConfig):
	"""
	Configuration class for the Bicephalous classifier.
	Stores the architecture and training-related parameters required to
	initialize a Bicephalous model, including the backbone transformer,
	output label dimensions, and quantization settings.
	"""
	def __init__(
			self, backbone_name:str|None=None, num_labels:int=0,
			num_coarse_labels:int=0, num_fine_labels:int=0,
			enable_quantization:bool=True, **kwargs
		) -> None:
		"""
		Initializes the model configuration.

		Arguments
		---------
		backbone_name : str | None, optional
			Name or identifier of the pretrained transformer backbone
			used by the model.

		num_labels : int, default=0
			Total number of labels in the dataset.

		num_coarse_labels : int, default=0
			Number of coarse-grained labels.

		num_fine_labels : int, default=0
			Number of fine-grained labels.

		enable_quantization : bool, default=True
			Whether model quantization should be enabled.

		**kwargs
			Additional configuration parameters passed to
			`transformers.PretrainedConfig`.
		"""
		super().__init__(**kwargs)
		self.backbone_name = backbone_name
		self.num_labels = num_labels
		self.num_coarse_labels = num_coarse_labels
		self.num_fine_labels = num_fine_labels
		self.enable_quantization = enable_quantization


class BicephalousClassifier(PreTrainedModel):
	"""
	Hierarchical classifier built on top of a pretrained transformer.

	The model consists of a transformer backbone followed by two
	classification heads for coarse and fine classification.

	The fine-grained head is optional and is instantiated only when
	fine-grained labels exist. The backbone may optionally be
	loaded using quantization and PEFT adapters.

	Notes
	-----
	The sequence representation is obtained from the final non-padding
	token in the input sequence rather than the first token.
	"""
	config_class = BicephalousConfig

	def __init__(self, config:BicephalousConfig, backbone:PreTrainedModel=None) -> None:
		"""
		Initializes the hierarchical classifier.

		Arguments
		---------
		config : BicephalousConfig
			Model configuration containing backbone, label, and
			quantization settings.

		backbone : transformers.PreTrainedModel | None, optional
			Pre-initialized transformer backbone. If None, the backbone is
			loaded from `config.backbone_name`.
		"""
		super().__init__(config)

		if backbone is None:
			tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
			backbone = self.load_backbone(config, tokenizer)

		self.backbone = backbone
		self.hidden_size = self.backbone.config.hidden_size
		self.dropout = nn.Dropout(0.1)
		self.num_labels = config.num_labels
		self.coarse_head = nn.Linear(self.hidden_size, config.num_coarse_labels)
		self.fine_head = nn.Linear(self.hidden_size, config.num_fine_labels) if config.num_fine_labels > 0 else None

		self.post_init()


	@classmethod
	def load_backbone(
		cls, config:BicephalousConfig, tokenizer:PreTrainedTokenizerBase
	) -> PreTrainedModel:
		"""
		Loads and configures the transformer backbone.

		Applies tokenizer-dependent vocabulary resizing and optionally
		enables quantization and PEFT-based fine-tuning.

		Arguments
		---------
		config : BicephalousConfig
			Model configuration.

		tokenizer : PreTrainedTokenizerBase
			Tokenizer associated with the backbone model.

		Returns
		-------
		backbone : PreTrainedModel
			Configured transformer backbone.
		"""
		if config.enable_quantization: bnb_config, peft_config = quantization_config()
		else: bnb_config, peft_config = (None, None)

		backbone = AutoModel.from_pretrained(
			config.backbone_name,
			quantization_config=bnb_config,
			attn_implementation="sdpa",
			dtype=torch.bfloat16,
			device_map="auto",
			trust_remote_code=True,
		)

		if len(tokenizer) != backbone.get_input_embeddings().num_embeddings:
			backbone.resize_token_embeddings(len(tokenizer))
		backbone.config.use_cache = False

		if config.enable_quantization:
			backbone = prepare_model_for_kbit_training(backbone)
			backbone = get_peft_model(backbone, peft_config)

		return backbone


	# @classmethod
	# def load_from_checkpoint(cls, ckpt_path:Path): # TODO Investigate whether this is even being used or not.
	# 	"""
	# 	Loads a saved model checkpoint.

	# 	Restores the model configuration, tokenizer, backbone, and model
	# 	weights from the specified checkpoint directory.

	# 	Arguments
	# 	---------
	# 	ckpt_path : Path
	# 		Path to the checkpoint directory.

	# 	Returns
	# 	-------
	# 	model, tokenizer : tuple[BicephalousClassifier, PreTrainedTokenizerBase]
	# 		Loaded model and associated tokenizer.
	# 	"""
	# 	config = BicephalousConfig.from_pretrained(ckpt_path)
	# 	tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
	# 	backbone = cls.load_backbone(config, tokenizer)
	# 	model = cls(config, backbone)

	# 	state_dict = torch.load(f"{ckpt_path}/pytorch_model.bin", map_location="cpu")
	# 	model.load_state_dict(state_dict, strict=False)

	# 	return model, tokenizer


	def gradient_checkpointing_enable(self) -> None:
		"""
		Enables gradient checkpointing for the transformer backbone.
		"""
		self.backbone.gradient_checkpointing_enable()


	def forward(
		self, input_ids:Tensor, attention_mask:Tensor,
		output_hidden_states:bool=False, return_dict:bool=True, **kwargs
	) -> dict[str, Tensor]:
		"""
		Performs a forward pass through the model.

		Arguments
		---------
		input_ids : Tensor
			Input token IDs.
			Shape: (batch_size, 512)

		attention_mask : Tensor
			Attention mask indicating valid tokens.
			Shape: (batch_size, 512)

		output_hidden_states : bool, default=False
			Whether to include the sequence representation in the output.

		return_dict : bool, default=True
			Included for compatibility with the Hugging Face model API.

		**kwargs
			Additional arguments forwarded to the backbone model.

		Returns
		-------
		dict[str, torch.Tensor | None]
			Dictionary containing:

			coarse_logits
				Coarse-grained classification logits.
				Shape: (batch_size, 2)

			fine_logits
				Fine-grained classification logits.
				Shape: (batch_size, 2)
				or None when no fine head is configured.

			cls_emb
				Sequence representation extracted from the final
				non-padding token.
				Shape: (batch_size, hidden_size)

				Returned only when `output_hidden_states=True`.
		"""
		outputs = self.backbone(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=output_hidden_states,
			return_dict=True,
			**kwargs
		)

		last_hidden = outputs.last_hidden_state
		seq_lengths = attention_mask.sum(dim=1) - 1
		cls_emb = last_hidden[ # TODO: Make this clearer.
			torch.arange(last_hidden.size(0), device=last_hidden.device),
			seq_lengths
		]  # NOT [CLS] in the strictest sense. Named for convenience.
		cls_dropout = self.dropout(cls_emb)
		coarse_logits = self.coarse_head(cls_dropout)
		fine_logits = self.fine_head(cls_dropout) if self.fine_head is not None else None

		return {
			"coarse_logits": coarse_logits,
			"fine_logits": fine_logits,
			"cls_emb": cls_emb if output_hidden_states else None
		}


class BicephalousTrainer(Trainer):
	"""
	Custom trainer for hierarchical classification.

	Extends the Hugging Face Trainer by:
	- Computing class-balanced loss weights from the training set.
	- Optimizing over a combined coarse- and fine-grained loss.
	- Logging coarse and fine loss components separately.
	- Reconstructing final class probabilities during evaluation.

	The trainer supports both binary and three-class hierarchical
	classification settings.
	"""
	def __init__(self, *args, **kwargs):
		"""
		Initializes the trainer.
		Computes class weights from the training dataset and prepares the
		loss configuration used during optimization.

		Side Effects
		------------
		- Computes coarse-grained class weights.
		- Computes fine-grained class weights.
		- Stores the training device.
		- Initializes the loss weighting factor `lam`.
		"""
		super().__init__(*args, **kwargs)
		self.device = self.args.device
		self.coarse_weights = ComputeWeightsAndLosses.compute_weights(self.train_dataset["coarse_label"]).to(self.device)
		self.fine_weights = ComputeWeightsAndLosses.compute_weights(self.train_dataset["fine_label"], fine_flag=True).to(self.device)
		self.lam = 1.0


	def log(self, logs:dict[str, float], *args, **kwargs):
		"""
		Logs training and evaluation metrics.

		Adds the most recently computed coarse and fine loss values to the
		log dictionary before forwarding it to the parent trainer.

		Parameters
		----------
		logs : dict[str, float]
			Metrics to be logged.

		Returns
		-------
		None
		"""
		if hasattr(self, "_last_coarse_loss"):
			logs["coarse_loss"] = self._last_coarse_loss
		if hasattr(self, "_last_fine_loss"):
			logs["fine_loss"] = self._last_fine_loss

		super().log(logs, *args, **kwargs)


	def compute_loss(
		self, model:BicephalousClassifier, inputs:dict[str, Tensor],
		return_outputs:bool=False, num_items_in_batch:int=None
	):
		"""
		Computes the hierarchical training loss.
		Calculates coarse- and fine-grained classification losses and
		combines them according to the weighting factor `lam`.

		Arguments
		---------
		model : BicephalousClassifier
			Model being optimized.

		inputs : dict[str, Tensor]
			Batch of training inputs containing:
			- input_ids
			- attention_mask
			- coarse_label
			- fine_label
			- labels

		return_outputs : bool, default=False
			Whether to return model outputs together with the loss.

		num_items_in_batch : int | None, optional
			Batch size information provided by the trainer.

		Returns
		-------
		total_loss : Tensor
			Total training loss.

		outputs : dict
			Model outputs. Returned when `return_outputs=True`.
		"""
		coarse_label = inputs.pop("coarse_label").to(self.device)
		fine_label = inputs.pop("fine_label")
		outcomes = inputs.pop("labels").to(self.device)
		outputs = model(**inputs)

		# print(f"Coarse weights: {self.coarse_weights}\nFine weights: {self.fine_weights}")
		total_loss, coarse_loss, fine_loss = ComputeWeightsAndLosses.compute_loss(
			outputs=outputs,
			coarse_label=coarse_label,
			fine_label=fine_label,
			lam=self.lam,
			coarse_weights=self.coarse_weights,
			fine_weights=self.fine_weights
		)

		if self.state.global_step % self.args.logging_steps == 0:
			self._last_coarse_loss = coarse_loss.detach().item()
			self._last_fine_loss = fine_loss.detach().item()

		return (total_loss, outputs) if return_outputs else total_loss


	def prediction_step(
		self, model:BicephalousClassifier, inputs:dict[str, Tensor],
		prediction_loss_only:bool,
		ignore_keys:list[str] | None=None
	) -> tuple[None, Tensor, Tensor]:
		"""
		Performs a prediction step during evaluation.

		Converts model outputs into final class probabilities suitable for
		metric computation. For three-class classification, the probabilities
		are reconstructed from the hierarchical coarse and fine predictions.
		For binary classification, the coarse probabilities are used directly.

		Arguments
		---------
		model : BicephalousClassifier
			Model used for inference.

		inputs : dict[str, Tensor]
			Evaluation batch.

		prediction_loss_only : bool
			Included for compatibility with the Trainer API.

		ignore_keys : list[str] | None, optional
			Keys to ignore during prediction.

		Returns
		-------
		tuple[None, Tensor, Tensor]
			Tuple containing:
			- loss (always None)
			- prediction probabilities
			- ground-truth labels
		"""
		inputs = {
			k: torch.tensor(v).to(self.device) if not isinstance(v, torch.Tensor)
			else v.to(self.device)
			for k, v in inputs.items()
		}
		coarse_label = inputs.pop("coarse_label", None).to(self.device)
		fine_label = inputs.pop("fine_label", None).to(self.device)
		labels = inputs.pop("labels", None).to(self.device)

		with torch.no_grad():
			outputs = model(**inputs)
			coarse_logits = outputs["coarse_logits"]
			fine_logits = outputs["fine_logits"]

			coarse_prob = torch.softmax(coarse_logits, dim=-1)
			if model.num_labels == 3:
				"""
				For three-class classification, final probabilities are reconstructed
				as:
				- P(dismissed) = P(coarse=negative)
				- P(allowed) = P(coarse=positive) * P(fine=allowed)
				- P(partly_allowed) = P(coarse=positive) * P(fine=partly_allowed)

				This produces a probability distribution over the original label
				space.
				"""
				fine_prob = torch.softmax(fine_logits, dim=-1)
				final_probs = torch.zeros(coarse_logits.size(0), 3, device=self.device)

				# Schema defined in `load_data.py`
				final_probs[:, 0] = coarse_prob[:,0] 				  # dismissed
				final_probs[:, 1] = coarse_prob[:,1] * fine_prob[:,1] # allowed
				final_probs[:, 2] = coarse_prob[:,1] * fine_prob[:,0] # partly allowed

			else:
				final_probs = coarse_prob

		return (None, final_probs, labels)
