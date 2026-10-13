import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

import json, tiktoken
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt

from config import DATASET, TOOLS


def compute_lengths(data_filepath:Path) -> tuple[list[int], list[int], list[int]]:
	"""
	Computes the total number of characters, words, and tokens
	in each judgment.

	Arguments
	---------
	data_filepath : Path
		Filepath to the dataset.

	Returns
	-------
	char_count : list[int]
		List of character counts per judgment.

	word_count : list[int]
		List of word counts per judgment.

	token_count : list[int]
		List of token counts per judgment.
	"""
	char_count, word_count, token_count = [], [], []
	gpt_enc = tiktoken.encoding_for_model("gpt-4")
	skipped_counter = 0
	judgment_counter = 0

	with open(data_filepath, 'r') as df:
		# for line in tqdm(df, desc="Reading"):
		# 	data = json.loads(line)
		# 	judgment_text = data["judgment"]
		# 	if len(judgment_text.split()) < 50: 
		# 		skipped_counter += 1
		# 		continue

		judgments = [json.loads(line) for line in df]
		for judgment in tqdm(judgments):
			judgment_text = judgment["judgment"]
			if len(judgment_text.split()) < 50: 
				skipped_counter += 1
				continue

			char_count.append(len(judgment_text))
			word_count.append(len(judgment_text.split(" ")))
			token_count.append(len(gpt_enc.encode(judgment_text)))
			judgment_counter += 1

	print(skipped_counter, judgment_counter)
	return char_count, word_count, token_count


def plot_distributions(counts: pd.DataFrame) -> None:
	fig, axes = plt.subplots(
		1, 2,
		figsize=(12, 4.5),
		dpi=120,
		constrained_layout=True,
	)

	for ax, column, label in zip(axes,
		["word_count", "token_count"],
		["Word count", "Token count"],
	):
		values = counts[column].dropna()

		bins = np.logspace(
			np.log10(values.min()),
			np.log10(values.max()),
			50,
		)

		ax.hist(values, bins=bins)

		ax.set_xscale("log")
		ax.set_title(f"{label} distribution", fontsize=12)
		ax.set_xlabel(label, fontsize=10)
		ax.set_ylabel("Frequency", fontsize=10)
		ax.tick_params(axis="both", labelsize=9)

	fig.savefig("count_distribution.png", dpi=300, bbox_inches="tight")
	plt.show()


def plot_graphs(counts:pd.DataFrame) -> None:
	"""
	Plots word and token length distribution histographs.
	Displays 4 subplots for different ranges of distribution.

	Arguments
	---------
	counts : pd.DataFrame
		Dataframe containing character, word, and token count.

	Returns
	-------
	None
	"""
	subplot_size = 4
	fig, axes = plt.subplots(subplot_size, 2, figsize=(15,5), dpi=300)
	lim = [10_000, 50_000, 1_00_000, 1_00_000]

	for i in range(subplot_size):
		if i == 0:
			axes[i, 0].hist(counts["word_count"][counts["word_count"] < lim[i]], bins=50)
			axes[i, 0].set_title(label=f"Word count < {lim[i]}")
			axes[i, 0].set_xlabel("Length distribution")
			axes[i, 0].set_ylabel("Frequency")

			axes[i, 1].hist(counts["token_count"][counts["token_count"] < lim[i]], bins=50)
			axes[i, 1].set_title(label=f"Token count < {lim[i]}")
			axes[i, 1].set_xlabel("Length distribution")
			axes[i, 1].set_ylabel("Frequency")

		elif i == 3:
			axes[i, 0].hist(counts["word_count"][counts["word_count"] > lim[i]], bins=50)
			axes[i, 0].set_title(label=f"Word count > {lim[i]}")
			axes[i, 0].set_xlabel("Length distribution")
			axes[i, 0].set_ylabel("Frequency")

			axes[i, 1].hist(counts["token_count"][counts["token_count"] > lim[i]], bins=50)
			axes[i, 1].set_title(label=f"Token count > {lim[i]}")
			axes[i, 1].set_xlabel("Length distribution")
			axes[i, 1].set_ylabel("Frequency")

		else:
			axes[i, 0].hist(counts["word_count"][counts["word_count"].apply(lambda x: x in range(lim[i-1], lim[i]))], bins=50)
			axes[i, 0].set_title(label=f"Word count in range ({lim[i-1]}, {lim[i]})")
			axes[i, 0].set_xlabel("Length distribution")
			axes[i, 0].set_ylabel("Frequency")

			axes[i, 1].hist(counts["token_count"][counts["token_count"].apply(lambda x: x in range(lim[i-1], lim[i]))], bins=50)
			axes[i, 1].set_title(label=f"Token count in range ({lim[i-1]}, {lim[i]})")
			axes[i, 1].set_xlabel("Length distribution")
			axes[i, 1].set_ylabel("Frequency")

	plt.tight_layout()
	plt.show()


def main() -> None:
	"""
	Loads the data, saves the character, word, and token counts
	to a CSV file, and plots graphs for them.
	"""
	data_filepath = DATASET / "SCIJuDAC.jsonl"
	count_output_filename = TOOLS / "dataset_analysis" / "count.csv"

	if count_output_filename.exists():
		print("count.csv already exists.")
		counts = pd.read_csv(count_output_filename)
		print(counts.describe())

	else:
		char_count, word_count, token_count = compute_lengths(data_filepath)

		counts = pd.DataFrame(columns=["char_count", "word_count", "token_count"])
		counts["char_count"] = char_count
		counts["word_count"] = word_count
		counts["token_count"] = token_count
		counts.to_csv(count_output_filename, index=False)

		print(counts.describe())

	plot_distributions(counts)


if __name__ == '__main__':
	main()