import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

import json, tiktoken
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt

from config import DATASET, TOOLS


def plot_graphs(counts:pd.DataFrame):
	# x = range(len(count))
	# y = count[count < 50000]
	# y_log = np.log10(y)
	# plt.hist(y, 50)
	subplot_size = 4
	fig, axes = plt.subplots(subplot_size, 2, figsize=(15,5))
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


def main():
	data_file = DATASET / "SCIJuDAC.jsonl"
	count_output_filename = TOOLS / "count.csv"

	if count_output_filename.exists():
		counts = pd.read_csv(count_output_filename)

	else:
		char_count, word_count, token_count = [], [], []
		gpt_enc = tiktoken.encoding_for_model("gpt-4")

		with open(data_file, 'r') as df:
			for line in tqdm(df):
				data = json.loads(line)
				judgment = data["judgment"]
				judgment_text = ""
				for d in judgment: judgment_text += d["text"]
				char_count.append(len(judgment_text))
				word_count.append(len(judgment_text.split(" ")))
				token_count.append(len(gpt_enc.encode(judgment_text)))

		counts = pd.DataFrame(columns=["char_count", "word_count", "token_count"])
		counts["char_count"] = char_count
		counts["word_count"] = word_count
		counts["token_count"] = token_count
		counts.to_csv(count_output_filename, index=False)

	for column in counts.columns:
		plot_graphs(counts[column], column)