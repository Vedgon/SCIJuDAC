"""
Helper functions to load and tokenize the dataset.
Additional functions do tasks like merging the two
similar labels into one for ablation study,
casting the text labels to numbers, and mapping
the labels for fine-grained classification for
SCIJuDAC.
"""
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import PreTrainedTokenizerBase
from collections import defaultdict

def merge_labels(
		batch:Dataset
	) -> Dataset:
	"""
	Helper function to merge the `allowed` and `partly
	allowed` labels in the SCIJuDAC dataset for the
	ablation study. Called when `collapsed_label`
	parameter is set to "y".

	Arguments
	---------
	batch : Dataset
		Batch of dataset rows.

	Returns
	-------
	batch : Dataset
		Batch of dataset rows.
	"""
	for i, label in enumerate(batch["outcome"]):
		if label in ["allowed", "partly allowed"]:
			label = "allowed"
			batch["outcome"][i] = label

	return batch


def cast_labels(self, dataset:Dataset) -> Dataset:
	"""
	Casts string labels from the dataset to integers.

	Arguments
	---------
	self : Namespace object of instance attributes for the calling model.
		Contains model's instance attributes.

	dataset : Dataset
		Dataset object instance loaded from the disk.

	Returns
	-------
	casted_dataset : Dataset
		Dataset object instance with labels converted to integers.
	"""
	if self.collapse_labels == "y":
		print("Collapsing \"allowed\" and \"partly allowed\" into \"allowed\".")
		dataset = dataset.map(merge_labels,
			desc="Merging allowed and partly_allowed", batched=True
		)
		self.num_labels = 2
	else:
		outcome_labels = ["dismissed", "allowed", "partly allowed"]

	if self.num_labels == 2: outcome_labels = ["dismissed", "allowed"]

	label_feature = ClassLabel(names=outcome_labels)
	casted_dataset = dataset.cast_column("outcome", label_feature)

	return casted_dataset


def derive_hierarchical_labels(
		batch:Dataset, num_labels:int
	) -> dict[str:list, str:list]:
	"""
	Derives the labels in terms of coarse and fine labels.
	Populates the `fine` list only when the number of labels is 3
	and the outcome label is 1 or 2. Fills the `fine` list with
	a dummy value (-100) otherwise. The returned dict is added
	as columns for each row.

	Arguments
	---------
	batch : Dataset
		Batch of dataset rows.

	num_labels : int
		Number of labels present in the dataset.

	Returns
	-------
	dict[str:list, str:list]
		Dictionary to add as columns for each batch of dataset rows.
	"""
	coarse, fine = [], []

	for outcome in batch["outcome"]:
		if num_labels == 3:
			if outcome in [1, 2]:
				coarse.append(1)
				fine.append(1 if outcome == 1 else 0)
			else:
				coarse.append(0)
				fine.append(-100)

		else:
			if outcome == 0:
				coarse.append(0)
				fine.append(-100)
			else:
				coarse.append(1)
				fine.append(-100)

	return {"coarse_label": coarse, "fine_label": fine}


def create_input_ids(
		batch:Dataset, tokenizer:PreTrainedTokenizerBase
	) -> Dataset:
	"""
	Tokenizes the data to standard HF format. Adds the 
	coarse and fine labels to the tokenized output dictionary.

	Arguments
	---------
	batch : Dataset
		Batch of dataset rows.

	tokenizer : PreTrainedTokenizerBase
		Associated tokenizer instance loaded during model loading.


	Returns
	-------
	outputs : Dataset
		Batch of rows of tokenized data. Added back to the dataset.

	"""
	tokenizer.truncation_side = "left"
	outputs = tokenizer(
		batch["judgment"],
		max_length=512,
		truncation=True,
		return_attention_mask=True,
		padding=False,
	)

	outputs["coarse_label"] = [batch["coarse_label"][i] for i in range(len(outputs["input_ids"]))]
	outputs["fine_label"] = [batch["fine_label"][i] for i in range(len(outputs["input_ids"]))]
	outputs["labels"] = [batch["outcome"][i] for i in range(len(outputs["input_ids"]))]

	return outputs


def load_and_tokenize_data(self, tokenizer:PreTrainedTokenizerBase) -> tuple[Dataset, Dataset, Dataset]:
	"""
	Loads the dataset from the disk, splits it into train, validation, and test
	sets, and tokenizes it via the model's tokenizer.
	Arguments
	---------
	self : Namespace object of instance attributes for the calling model.
		Contains model's instance attributes.

	tokenizer : PreTrainedTokenizerBase
		Associated tokenizer instance loaded during model loading.

	Returns
	-------
	train_dataset : Dataset
		Train split of the dataset.

	valid_dataset : Dataset
		Validation split of the dataset.

	test_dataset : dataset
		Test split of the dataset.
	"""
	dataset = load_dataset("json", data_files=str(self.dataset_path), split="train", download_mode="force_redownload")
	# dataset = dataset.select(range(40))

	dataset = cast_labels(self, dataset)
	dataset = dataset.map(
		derive_hierarchical_labels,
		batched=True,
		num_proc=4,
		fn_kwargs={"num_labels": self.num_labels},
		desc="Deriving labels"
	)

	dataset.select_columns(["file_ID", "judgment", "outcome", "coarse_label", "fine_label"])
	# print(dataset["fine_label"])

	dataset = dataset.train_test_split(test_size=0.3, seed=self.seed)
	val_test_dataset = dataset["test"].train_test_split(test_size=0.66, seed=self.seed)
	split_dataset = DatasetDict({
		"train" : dataset["train"],
		"valid" : val_test_dataset["train"],
		"test" : val_test_dataset["test"]
	})

	tokenized_dataset = split_dataset.map(
		create_input_ids,
		batched=True,
		remove_columns=split_dataset["train"].column_names,
		num_proc=4,
		load_from_cache_file=False,
		fn_kwargs={"tokenizer": tokenizer},
		desc="Tokenizing"
	)

	tokenized_dataset.set_format(
		type="torch",
		columns=["input_ids", "attention_mask", "coarse_label", "fine_label", "labels"]
	)

	train_dataset = tokenized_dataset["train"]
	valid_dataset = tokenized_dataset["valid"]
	test_dataset = tokenized_dataset["test"]
	# print(train_dataset[0])

	return train_dataset, valid_dataset, test_dataset
