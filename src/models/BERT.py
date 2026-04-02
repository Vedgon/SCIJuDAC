import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TrainingArguments,
	EarlyStoppingCallback,
	Trainer
)
import logging, torch

from config import CHECKPOINTS_ROOT
from training import load_and_tokenize_data, load_checkpoint_from_disk, ComputeMetrics, LoggingCallback

logger = logging.getLogger("LJP_exp")


class BERT():
	def __init__(
			self, dataset_path:Path, num_labels:int,
			model_name:str, batch_size:int,
			epochs:int, seed:int,
			all_toks:int, collapse_labels:int):

		self.dataset_path = dataset_path
		self.num_labels = num_labels
		self.model_name = model_name
		self.batch_size = batch_size
		self.epochs = epochs
		self.seed = seed
		self.all_toks = all_toks
		self.collapse_labels = collapse_labels


	def train_model(self, model, tokenizer, train_dataset, valid_dataset):
		device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
		model.to(device)

		if self.all_toks: toks = "all_toks"
		else: toks = "last_toks"
		dataset_name = self.dataset_path.stem
		model_name = self.model_name.split("/")[-1]
		if self.collapse_labels: checkpoint_output_path = CHECKPOINTS_ROOT / dataset_name / "ablation" / toks / model_name
		else: checkpoint_output_path = CHECKPOINTS_ROOT / dataset_name / toks / model_name
		checkpoint_output_path.mkdir(parents=True, exist_ok=True)

		max_grad_norm = 1.0

		training_args = TrainingArguments(
			fp16=True,
			num_train_epochs=self.epochs,
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size,
			dataloader_num_workers=4,

			max_grad_norm=max_grad_norm,
			gradient_accumulation_steps=4,
			weight_decay=0.01,
			seed=self.seed,
			data_seed=self.seed,

			learning_rate=2e-5,
			warmup_ratio=0.06,
			lr_scheduler_type="cosine",

			logging_strategy="steps",
			logging_steps=300,

			eval_strategy="steps",
			eval_steps=1500,

			output_dir=checkpoint_output_path,
			save_strategy="steps",
			save_steps=1500,
			save_total_limit=2,

			load_best_model_at_end=True,
			metric_for_best_model="f1_weighted",
			greater_is_better=True,
			remove_unused_columns=False
		)

		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
			tokenizer=tokenizer,
			compute_metrics=ComputeMetrics(valid_dataset.with_format(None), self.all_toks),
			callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback(logger)]
		)

		trainer = load_checkpoint_from_disk(checkpoint_output_path, trainer, logger)

		return trainer


	def run(self):
		model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
		tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		model.gradient_checkpointing_enable()
		train_dataset, valid_dataset, test_dataset = load_and_tokenize_data(self, tokenizer)
		trainer = BERT.train_model(self, model, tokenizer, train_dataset, valid_dataset)
		predictions = trainer.predict(test_dataset)
		metrics = ComputeMetrics(test_dataset, self.all_toks)(predictions)

		logger.info("For test dataset:")
		for metric in metrics.keys():
			logger.info(f"{metric}: {metrics[metric]}")
