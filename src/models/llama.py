import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	DataCollatorWithPadding,
	TrainingArguments,
	EarlyStoppingCallback,
	Trainer
)
from peft import get_peft_model
import logging, os
from dotenv import load_dotenv

from config import CHECKPOINTS_ROOT
from training import (
	load_and_tokenize_data,
	model_config,
	load_checkpoint_from_disk,
	ComputeMetrics,
	LoggingCallback
)

logger = logging.getLogger("LJP_exp")


class Llama():
	def __init__(self, dataset_path:Path, num_labels:int,
				model_name:str, batch_size:int,
				epochs:int, seed:int,
				all_toks:int, collapse_labels:int
	):
		self.dataset_path = dataset_path
		self.num_labels = num_labels
		self.model_name = model_name
		self.batch_size = batch_size
		self.epochs = epochs
		self.seed = seed
		self.all_toks = all_toks
		self.collapse_labels = collapse_labels

		load_dotenv()
		self.HF_token = os.environ["HF_TOKEN"]


	def train_model(self, model, tokenizer, train_dataset, valid_dataset):
		data_collator = DataCollatorWithPadding(
			tokenizer=tokenizer,
			padding=True,
		)

		if self.all_toks: toks = "all_toks"
		else: toks = "last_toks"
		dataset_name = self.dataset_path.stem
		model_name = self.model_name.split("/")[-1]
		if self.collapse_labels: checkpoint_output_path = CHECKPOINTS_ROOT / dataset_name / "ablation" / toks / model_name
		else: checkpoint_output_path = CHECKPOINTS_ROOT / dataset_name / toks / model_name
		checkpoint_output_path.mkdir(parents=True, exist_ok=True)

		training_args = TrainingArguments(
			output_dir=checkpoint_output_path,

			num_train_epochs=self.epochs,
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size,
			gradient_accumulation_steps=4,

			learning_rate=2e-5,
			lr_scheduler_type="cosine",
			warmup_ratio=0.05,
			weight_decay=0.01,
			max_grad_norm=1.0,

			fp16=False,
			bf16=False,

			logging_strategy="steps",
			logging_steps=500,
			eval_strategy="steps",
			save_strategy="steps",
			eval_steps=1500,
			save_steps=1500,
			save_total_limit=2,

			load_best_model_at_end=True,
			metric_for_best_model="f1_weighted",
			greater_is_better=True,

			dataloader_num_workers=4,
			dataloader_pin_memory=True,
			report_to="none",
			remove_unused_columns=False,
		)

		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
			data_collator=data_collator,
			compute_metrics=ComputeMetrics(valid_dataset.with_format(None), self.all_toks),# build_compute_metrics(valid_dataset, self.all_toks),
			callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback(logger)]
		)

		trainer = load_checkpoint_from_disk(checkpoint_output_path, trainer, logger)

		return trainer


	def run(self):
		tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, token=self.HF_token)
		tokenizer.pad_token = tokenizer.eos_token

		train_dataset, valid_dataset, test_dataset = load_and_tokenize_data(self, tokenizer)
		# print(train_dataset, valid_dataset, test_dataset)
		# print(len(train_dataset), len(valid_dataset), len(test_dataset))
		# print(len(set(valid_dataset["file_id"])))
		bnb_config, peft_config = model_config()

		base_model = AutoModelForSequenceClassification.from_pretrained(
			self.model_name,
			quantization_config=bnb_config,
			device_map="auto",
			attn_implementation="sdpa",
			trust_remote_code=True,
			num_labels=self.num_labels,
			token=self.HF_token
		)
		base_model.config.pad_token_id = tokenizer.pad_token_id
		base_model.config.use_cache = False

		model = get_peft_model(base_model, peft_config)
		model.config.pad_token_id = tokenizer.eos_token_id
		model.resize_token_embeddings(len(tokenizer))
		model.gradient_checkpointing_enable()
		trainer = Llama.train_model(self, model, tokenizer, train_dataset, valid_dataset)
		metrics = trainer.evaluate(eval_dataset=test_dataset)

		logger.info("For test dataset:")
		for metric in metrics.keys():
			logger.info(f"{metric}: {metrics[metric]}")
