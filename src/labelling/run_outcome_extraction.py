from pathlib import Path
import sys
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from collections import Counter
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from config import PARAWISE_DATA, LABELLING
from extract_outcome_label import label_extraction_from_text, label_extraction_from_outcome


def plot_label_dist(labels:list[str]) -> None:
	"""
	Plots graphs for label distribution.

	Arguments
	---------
	labels : list[str]
		List of labels extracted from judgments

	Returns
	-------
	None
	"""
	counter = Counter(labels).most_common()
	label_count = {label: count for label, count in counter if count > 6}

	for key in label_count: print(f"{key}: {label_count[key]}")

	x = [key for key in label_count.keys()]
	y = [value for value in label_count.values()]

	for i in range(len(x)): plt.text(i-0.1, y[i]+500, y[i])
	plt.bar(x, y, color="green")
	plt.xticks(rotation=60, ha="center")
	plt.grid(axis='y', alpha=0.3)
	plt.xlim(-0.5, len(label_count)-0.5)
	plt.tight_layout()
	plt.show()


def main() -> None:
	"""
	Gets all the judgment texts and extracts outcome labels from it 
	via two label_extraction functions. Also plots distribution of labels.
	"""
	judgment_data_root = PARAWISE_DATA
	folders = sorted(folder for folder in judgment_data_root.iterdir() if folder.is_dir())
	labels = []
	outcomes = []

	for folder in tqdm(folders):
		# if folder.name != "1950": continue
		judgment_files = sorted(file for file in folder.iterdir() if file.is_file())

		for file in judgment_files:
			# if file.name not in ["1.csv"]: continue
			with open(file, 'r', encoding="utf-8") as jf:
				judgment_text = pd.read_csv(jf, keep_default_na=False)
				decision = judgment_text.loc[judgment_text.index[-1], "text"]

				label = label_extraction_from_text(judgment_text)
				if label == "": label = label_extraction_from_outcome(decision)

				outcomes.append((f"{file.parent.name}/{file.name}", label))
				labels.append(label)

	labels_output_path = LABELLING / "outcomes_output.csv"
	outcomes_df = pd.DataFrame(outcomes, columns=["filepath", "label"])
	outcomes_df.to_csv(labels_output_path, index=False)

	plot_label_dist(labels)


if __name__ == "__main__":
	main()
