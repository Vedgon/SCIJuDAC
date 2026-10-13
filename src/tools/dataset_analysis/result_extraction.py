import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

from pathlib import Path
from tqdm import tqdm
import re, json

from config.config import SEPARABILITY_AND_CALIBRATION_LOGS


def parse_names(log_filepath:Path) -> tuple[str, str, str]:
	"""
	Extracts the names of the model, dataset, and document
	span (log file identifier; all tokens or last tokens)
	from the log filename.

	Arguments
	---------
	log_filepath : Path
		File path to the log file.

	Returns
	-------
	doc_size : str
		Log file identifier.

	model : str
		Name of the model.

	dataset : str
		Name of the dataset.
	"""
	keys = log_filepath.name.split("_")
	doc_size = f"{keys[0]}_{keys[1]}"
	if len(keys) == 6 and keys[2] != "lc":
		model = keys[2]
		dataset = f"{keys[3]}_{keys[4]}"
	elif len(keys) == 5:
		model = keys[2]
		dataset = keys[3]
	elif len(keys) == 6 and keys[2] == "lc":
		model = keys[3]
		dataset = f"{keys[4]}_{keys[2]}"

	return doc_size, model, dataset


def extract_margins(logs:list[str]) -> tuple[list, list]:
	"""
	Extracts the list of margin scores for correct and incorrect
	top-2 logit margin values.

	Arguments
	---------
	logs : list[str]
		List of log entries.

	Returns
	-------
	correct_margins : list
		Dictionary of list of correct margin values.

	incorrect_margins : list
		Dictionary of list of incorrect margin values.
	"""
	correct_margins = []
	incorrect_margins = []
	pattern = r"\d\.\d+"

	for i, margin_log in enumerate(logs):
		margin_data = re.search(pattern, margin_log)
		if margin_data: correct_margins.append(float(margin_data.group()))
		else: break
	
	for margin_log in logs[i+1:]:
		margin_data = re.search(pattern, margin_log)
		if margin_data: incorrect_margins.append(float(margin_data.group()))
		else: break

	return correct_margins, incorrect_margins


def extract_margin_scores(logs:list[str], idx:int) -> dict[dict]:
	"""
	Extracts the different scores related to top-2 logit margin
	values.

	Arguments
	---------
	logs : list
		List of log entries.

	idx : int
		Index from where the log entries related to margin values start.

	Returns
	-------
	logit_scores : dict[dict]
		List of dictionaries containing the logit margin score values.
	"""
	keys = {
		1: "Mean Margin", 2: "Mean Correct Margin",
		3: "Mean Incorrect Margin", 4: "Correct Margins",
		5: "Incorrect Margins"
	}
	logit_scores = dict.fromkeys(keys.values())

	for i, margin_score_log in enumerate(logs[idx+1:], start=idx+1):
		if margin_score_log[33:-1] == "Fine logits margin scores": break
		for key, value in keys.items():
			search = re.search(value, margin_score_log)
			if search and key == 1:
				mean_margin = float(re.search(r"\d\.\d+", logs[i]).group())
				mean_correct_margin = float(re.search(r"\d\.\d+", logs[i+1]).group())
				mean_incorrect_margin = float(re.search(r"\d\.\d+", logs[i+2]).group())

				logit_scores[keys[1]] = mean_margin
				logit_scores[keys[2]] = mean_correct_margin
				logit_scores[keys[3]] = mean_incorrect_margin

			elif search and key == 4:
				correct_margins, incorrect_margins = extract_margins(logs[i+1:])
				logit_scores[keys[4]] = correct_margins
				logit_scores[keys[5]] = incorrect_margins

	return logit_scores


def print_scores(scores:dict) -> None:
	"""
	Helper function to print all the scores extracted from the logs.

	Arguments
	---------
	scores : dict
		Dictionary containing all the scores.

	Returns
	-------
	None
	"""
	for key, value in scores.items():
		if isinstance(value, float): print(f"{key}: {value}") # Brier, ECE, Silhouette
		elif isinstance(value, dict): # Coarse/Fine logit margins
			print(f"\n{key}")
			print(f"Length of {key}: {len(value)}")
			for k, v in value.items():
				if isinstance(v, float): print(f"{k}: {v}") # Mean logit margin scores
				else: print(f"{k}: {len(v)}") # All logit margin scores

	print("\n")


def extract_scores(logs:str) -> dict:
	"""
	Extracts the different scores in the log file.

	Arguments
	---------
	logs : str
		List of log entries.

	Returns
	-------
	scores : dict
		Dictionary containing all the scores as
		dictionaries/list of dictionaries.
	"""
	logs = logs.split("\n")
	keys = {1: "Brier score", 2: "Expected Calibration score",
		3: "Silhouette score", 4: "Coarse logits margin scores",
		5: "Fine logits margin scores",
		6: "Fine head not trained, so no fine logit margins.", 7: "Classification report"}

	scores = {}

	for i, log in enumerate(logs):
		for key, value in keys.items():
			search = re.search(value, log)
			if search and key in [1, 2, 3]:
				score = float(re.search(r"\d\.\d+", log).group())
				scores[value] = score

			elif search and key == 4:
				margin_scores = extract_margin_scores(logs, i)
				scores[value] = margin_scores

			elif search and key == 5:
				margin_scores = extract_margin_scores(logs, i)
				scores[value] = margin_scores

			if key in [6, 7]:
				continue

	# print_scores(scores) # Uncomment to get the printout for each log file's data.

	return scores


def main()-> None:
	"""
	Reads the log files from the `separability_and_calibration` folder,
	and extracts all the scores from it.
	"""
	results = {}
	log_folders = sorted(folder for folder in SEPARABILITY_AND_CALIBRATION_LOGS.iterdir() if folder.is_dir())
	for log_folder in tqdm(log_folders, position=0, desc="Extracting data"):
		log_files = sorted(file for file in log_folder.iterdir() if file.is_file())
		for log_filepath in tqdm(log_files, position=1, leave=False, desc=log_folder.stem):
			# if log_filepath.stem != "last_toks_llama_SCIJuDAC_18052026": continue
			with open(log_filepath, 'r') as lf:
				logs = lf.read()
				doc_size, model, dataset = parse_names(log_filepath)
				results.setdefault(doc_size, {}).setdefault(model, {})[dataset] = extract_scores(logs)

	results_path = Path(__file__).parent / "separability_and_calibration_results.json"
	with open(results_path, "w") as f:
		json.dump(results, f, indent=4)


if __name__ == '__main__':
	main()
