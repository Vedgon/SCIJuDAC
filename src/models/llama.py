"""
Model configuration for the Llama3.1-7B.
"""
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from transformers import (
	AutoTokenizer,
	DataCollatorWithPadding,
	TrainingArguments,
	EarlyStoppingCallback,
)
import logging, os
from typing import Literal
from datasets import Dataset
from dotenv import load_dotenv

from config import CHECKPOINTS_ROOT
from training import (
	load_and_tokenize_data,
	get_checkpoint_path,
	BicephalousConfig,
	BicephalousClassifier,
	BicephalousTrainer,
	ComputeMetrics,
	HierarchicalCollator,
	LoggingCallback
)

logger = logging.getLogger("LJP_exp")


class BicephalousLlama():
	"""
	Training pipeline for the BicephalousLlama classifier.
    Handles dataset loading, tokenizer initialization, model
    configuration, training, evaluation, and checkpoint loading.
	"""
	def __init__(
			self, model_name:str, dataset_path:Path,
			num_labels:int, batch_size:int,
			epochs:int, seed:int, collapse_labels:Literal["y", "n"],
			load_ckpt:Literal["y", "n"]
		):
		"""
		Initializes the configuration for model training. Configures the
		HF identifier for the model, dataset path, training hyperparameters,
		and experiment-specific flags.

		Arguments
		---------
		model_name : str
			Hugging Face model identifier used to initialize
			the pretrained transformer model.

		dataset_path : Path
			Path to the dataset.

		num_labels : int
			Number of labels corresponding to the dataset (ILDC:2, SCIJuDAC:3).

		batch_size : int
			Batch size for training

		epochs : int
			Number of epochs till which the model will be trained.

		seed : int
			Random seed value for reporducibility (defaults to 42).

		collapse_labels : Literal["y", "n"]
			Flag for collapsing the 3 labels into 2 for SCIJuDAC dataset.

		load_ckpt : Literal["y", "n"]
			Flag for loading the checkpoint saved mid-training.

		Returns
		-------
		None
		"""
		self.model_name = model_name
		self.dataset_path = dataset_path
		self.num_labels = num_labels
		self.batch_size = batch_size
		self.epochs = epochs
		self.seed = seed
		self.collapse_labels = collapse_labels
		self.load_checkpoint = load_ckpt

		load_dotenv()
		self.HF_token = os.environ["HF_TOKEN"]


	def train_model(self,
		model:BicephalousClassifier,
		tokenizer:AutoTokenizer,
		train_dataset:Dataset,
		valid_dataset:Dataset,
	) -> BicephalousTrainer:
		"""
		Method to set up the model and trainer configurations,
		to load the checkpoint, and to train the model.

		Arguments
		---------
		model : BicephalousClassifier
			Bicephalous model instance.

		tokenizer : AutoTokenizer
			Tokenizer loaded with the pretrained model.

		train_dataset : Dataset
			Train split of the loaded dataset.

		valid_dataset : Dataset
			Validation split of the loaded dataset.

		Returns
		-------
		trainer : BicephalousTrainer
			Configured instance of the trainer.
		"""
		checkpoint_path = get_checkpoint_path(
							CHECKPOINTS_ROOT, self.dataset_path,
							self.model_name, self.collapse_labels,
							self.load_checkpoint
						)
		# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True,)
		data_collator = HierarchicalCollator(pad_token_id=tokenizer.pad_token_id)

		training_args = TrainingArguments(
			fp16=False,
			bf16=True,
			num_train_epochs=self.epochs,
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size,

			max_grad_norm=1.0,
			gradient_accumulation_steps=4,
			weight_decay=0.01,
			seed=self.seed,
			data_seed=self.seed,

			learning_rate=2e-5,
			warmup_ratio=0.05,
			lr_scheduler_type="cosine",

			logging_strategy="steps",
			logging_steps=200,
			eval_strategy="steps",
			eval_steps=1000,

			output_dir=checkpoint_path,
			save_strategy="steps",
			save_steps=1000,
			save_total_limit=2,

			dataloader_num_workers=4,
			load_best_model_at_end=True,
			dataloader_pin_memory=True,
			report_to="none",
			metric_for_best_model="f1_weighted",
			greater_is_better=True,
			remove_unused_columns=False,
		)

		trainer = BicephalousTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
			processing_class=tokenizer,
			data_collator=data_collator,
			compute_metrics=ComputeMetrics(valid_dataset.with_format(None)),
			callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback(logger)]
		)

		if self.load_checkpoint == "y":
			logger.info("Resuming from checkpoint...")
			trainer.train(resume_from_checkpoint=checkpoint_path)

		else:
			trainer.train()

		return trainer


	def run(self):
		"""
		Method to initialize our bicephalous Llama model, load the dataset,
		and execute training.
		Loads the backbone model and the associated tokenizer from HF,
		then our Bicephalous version of the classifier, and the dataset,
		followed by the initiation of training. The final results are logger
		in the logger defined in `main.py`.

		Arguments
		---------
		None

		Returns
		-------
		None
		"""
		tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
			tokenizer.pad_token_id = tokenizer.eos_token_id

		config = BicephalousConfig(
			backbone_name=self.model_name,
			num_labels=self.num_labels,
			num_coarse_labels=2,
			num_fine_labels=2 if self.num_labels==3 else 0,
			enable_qunatization=True,
		)

		peft_backbone = BicephalousClassifier.load_backbone(config, tokenizer)
		model = BicephalousClassifier(config=config, backbone=peft_backbone)
		model.to(peft_backbone.device)

		model.config.pad_token_id = tokenizer.pad_token_id
		model.gradient_checkpointing_enable()

		train_dataset, valid_dataset, test_dataset = load_and_tokenize_data(self, tokenizer)

		trainer = self.train_model(model, tokenizer, train_dataset, valid_dataset)
		metrics = trainer.evaluate(eval_dataset=test_dataset)

		logger.info("For test dataset:")
		for metric in metrics.keys():
			logger.info(f"{metric}: {metrics[metric]}")
