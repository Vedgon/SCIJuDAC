from datasets import load_dataset, ClassLabel, DatasetDict

def merge_labels(example):
	for i, label in enumerate(example["outcome"]):
		if label in ["allowed", "partly allowed"]:
			label = "allowed"
			example["outcome"][i] = label

	return example


def extract_data(batch):
	judgment_texts = []
	for judgment in batch["judgment"]:
		judgment_text = [text for para in judgment for text in para["text"]]
		judgment_texts.append(" ".join(judgment_text))

	batch["judgment"] = judgment_texts

	return batch


def create_input_ids(data, all_toks, tokenizer):
	if all_toks:
		outputs = tokenizer(
			data["judgment"],
			max_length=512,
			truncation=True,
			return_attention_mask=True,
			padding=False,
			stride=412,
			return_overflowing_tokens=True,
		)

		sample_map = outputs.pop("overflow_to_sample_mapping")
		outputs["file_id"] = [data["file_ID"][i] for i in sample_map]
		outputs["labels"] = [data["outcome"][i] for i in sample_map]

	else:
		tokenizer.truncation_side = "left"
		outputs = tokenizer(
			data["judgment"],
			max_length=512,
			truncation=True,
			return_attention_mask=True,
			padding=False,
		)

		outputs["labels"] = [data["outcome"][i] for i in range(len(outputs["input_ids"]))]

	return outputs


def load_and_tokenize_data(self, tokenizer):
	dataset = load_dataset("json", data_files=str(self.dataset_path), split="train", download_mode="force_redownload")
	# dataset = dataset.select(range(10))

	if self.num_labels == 3 and self.collapse_labels:
		print("Collapsing \"allowed\" and \"partly allowed\" into \"allowed\".")
		dataset = dataset.map(merge_labels,
			desc="Merging allowed and partly_allowed",
			batched=True
		)
		self.num_labels = 2
	else:
		outcome_labels = ["dismissed", "allowed", "partly allowed"]

	if self.num_labels == 2: outcome_labels = ["dismissed", "allowed"]

	label_feature = ClassLabel(names=outcome_labels)
	dataset = dataset.cast_column("outcome", label_feature)

	# dataset = dataset.map(
	# 	extract_data,
	# 	desc="Extracting data",
	# 	batched=True
	# )
	dataset.select_columns(["file_ID", "judgment", "outcome"]) # Can this be optimized by not selecting file_ID?

	dataset = dataset.train_test_split(test_size=0.3, seed=self.seed)
	val_test_dataset = dataset["test"].train_test_split(test_size=0.66, seed=self.seed)
	split_dataset = DatasetDict({
		"train" : dataset["train"],
		"valid" : val_test_dataset["train"],
		"test" : val_test_dataset["test"]
	})

	# print(len(split_dataset["train"]["outcome"]))

	tokenized_dataset = split_dataset.map(
		create_input_ids,
		batched=True,
		remove_columns=split_dataset["train"].column_names,
		num_proc=4,
		load_from_cache_file=False,
		fn_kwargs={"all_toks": self.all_toks, "tokenizer": tokenizer},
		desc="Tokenizing"
	)

	tokenized_dataset.set_format(
		type="torch",
		columns=["input_ids", "attention_mask", "labels"]
	)

	train_dataset = tokenized_dataset["train"]
	valid_dataset = tokenized_dataset["valid"]
	test_dataset = tokenized_dataset["test"]

	return train_dataset, valid_dataset, test_dataset