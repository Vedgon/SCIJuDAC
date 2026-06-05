"""
Model configuration for the BERT-based models.
"""
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from transformers import (
	AutoModel,
	AutoTokenizer,
	TrainingArguments,
	EarlyStoppingCallback,
)
import logging, torch
from typing import Literal
from datasets import Dataset

from config import CHECKPOINTS_ROOT
from training import (
	load_and_tokenize_data,
	get_checkpoint_path,
	BicephalousClassifier,
	BicephalousTrainer,
	ComputeMetrics,
	LoggingCallback
)

logger = logging.getLogger("LJP_exp")


class BicephalousBERT():
	"""
	Training pipeline for the BicephalousBERT classifier.
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

		train_dataset : datasets.Dataset
			Train split of the loaded dataset.

		valid_dataset:datasets.Dataset
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
		device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
		model.to(device)

		training_args = TrainingArguments(
			fp16=True,
			num_train_epochs=self.epochs,
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size,
			dataloader_num_workers=4,

			max_grad_norm=1.0,
			gradient_accumulation_steps=4,
			weight_decay=0.01,
			seed=self.seed,
			data_seed=self.seed,

			learning_rate=2e-5,
			warmup_ratio=0.05,
			lr_scheduler_type="cosine",

			logging_strategy="steps",
			logging_steps=300,

			evaluation_strategy="steps",
			eval_steps=1000,

			output_dir=checkpoint_path,
			save_strategy="steps",
			save_steps=1000,
			save_total_limit=2,

			load_best_model_at_end=True,
			metric_for_best_model="f1_weighted",
			greater_is_better=True,
			remove_unused_columns=False
		)

		trainer = BicephalousTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
			tokenizer=tokenizer,
			compute_metrics=ComputeMetrics(valid_dataset.with_format(None)),
			callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback(logger)]
		)

		if self.load_checkpoint == "y":
			logger.info("Resuming from checkpoint...")
			trainer.train(resume_from_checkpoint=checkpoint_path)

		else:
			trainer.train()

		return trainer


	def run(self) -> None:
		"""
		Method to initialize our bicephalous BERT model, load the dataset,
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
		encoder = AutoModel.from_pretrained(self.model_name)
		model = BicephalousClassifier(encoder, self.num_labels)
		tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		model.gradient_checkpointing_enable()

		train_dataset, valid_dataset, test_dataset = load_and_tokenize_data(self, tokenizer)

		trainer = self.train_model(model, tokenizer, train_dataset, valid_dataset)
		predictions = trainer.predict(test_dataset)
		metrics = ComputeMetrics(test_dataset)(predictions)

		logger.info("For test dataset:")
		for key, metric in metrics.items():
			logger.info(f"{key}: {metric}")
