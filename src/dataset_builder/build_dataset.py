from pathlib import Path
import sys
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

import re, json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import pandas as pd
from tqdm import tqdm

from config import LABELLING, PARAWISE_DATA, DATASET
from verdict_detection import detect_verdict_text, remove_verdict

stop_words = set(stopwords.words("english")) - set(["i", "my", "of"])


def clean_text(paragraph:str) -> str:
	"""
	Performs standard NLP preprocessing (punctuation and abbreviation removal).

	Arguments
	---------
	paragraph : str
		A paragraph from a judgment text.

	Returns
	-------
	paragraph : str
		Returns a cleaned paragraph.
	"""
	paragraph = paragraph.lower() if isinstance(paragraph, str) else str(paragraph)
	paragraph = re.sub(r"``|''|“|”|‘|’", "\"", paragraph)
	paragraph = re.sub(r"\bapp[\.]+", "application", paragraph)
	paragraph = re.sub(r"pet[\.]+", "petition", paragraph)
	paragraph = re.sub(r"no\.", "number", paragraph)

	return paragraph


def tokenize_text(paragraph:str) -> list[str]:
	"""
	Performs sentence and word-level tokenization.
	
	Arguments
	---------
	paragraph : str
		A paragraph from a judgment text.

	Returns
	-------
	cleaned_sentences : list[str]
		Returns a list of tokenized sentences.
	"""
	sentences = sent_tokenize(paragraph)
	tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
	cleaned_sentences = []

	for tokenized_sentence in tokenized_sentences:
		sentence = " ".join([token for token in tokenized_sentence if token not in string.punctuation and token not in stop_words])
		sentence = re.sub(r"``|''|“|”|‘|’", "\"", sentence)
		cleaned_sentences.append(sentence)

	return cleaned_sentences


def prepare_data(filepath:Path) -> tuple[list[str], list[str]]:
	"""
	Preprocesses the judgment data per paragraph by ultimately removing
	the sentences containing verdict.

	Arguments:
	filepath : Path
		Path to the judgment file.

	Returns
	-------
	paragraphs : list[str]
		List of paragraphs with verdicts removed. Returns an empty list in
		case of error.

	labels : list[str]
		List of labels associated with each paragraph, if available. Returns
		an empty list if labels unavailable or in case of error.
	"""
	paragraphs = []
	judgment_data = pd.read_csv(filepath).iloc[:-1]

	# Sets the initial paragraph index to start verdict detection from.
	# Guesstimated heuristic, probably not universally accurate.
	if len(judgment_data) < 10: verdict_start_idx = 3
	elif len(judgment_data) in range(10, 16): verdict_start_idx = 5
	else: verdict_start_idx = 0

	if verdict_start_idx != 0:
		# Adding all the paragraphs before the first suspected verdict-bearing paragraph.
		for paragraph in judgment_data["text"][:verdict_start_idx]:
			paragraphs.append(paragraph)

	for paragraph in judgment_data["text"][verdict_start_idx:-1]:
		new_paragraph = ""
		cleaned_paragraph = clean_text(paragraph)
		tokenized_paragraph = tokenize_text(cleaned_paragraph)
		verdict_sentence = detect_verdict_text(tokenized_paragraph)
		# print(f"Verdict sentence: {verdict_sentence}")

		if verdict_sentence:
			new_paragraph = remove_verdict(filepath, cleaned_paragraph, verdict_sentence)
			# print(f"Paragraph after verdict removal: {new_paragraph}")
			if len(new_paragraph) > 1: paragraphs.append(new_paragraph)
			elif new_paragraph == "": return

		else:
			paragraphs.append(str(paragraph))

	return paragraphs


def main() -> None:
	"""
	Preprocesses the data by removing the sentences containing verdict.
	Offers choice of processing data with and without labels (stored separately).
	Skips judgments without a relevant outcome.
	"""
	parawise_data_root = PARAWISE_DATA

	folders = sorted(folder for folder in parawise_data_root.iterdir() if folder.is_dir())
	outcomes_filepath = LABELLING / "outcomes_output.csv"
	outcomes = pd.read_csv(outcomes_filepath)
	final_data = []

	for folder in tqdm(folders, position=0, desc="Processing data"):
		# if folder.name not in ["1950"]: continue
		files = sorted(file for file in folder.iterdir() if file.is_file())

		for file in tqdm(files, position=1, leave=False, desc=folder.name):
			# if file.stem not in ["26"]: continue
			preprocessed_data = {}
			key = f"{file.parent.name}/{file.name}"
			outcome = outcomes.loc[outcomes["filepath"] == key]["label"].squeeze()
			if outcome not in ["allowed", "dismissed", "partly allowed"]: continue # Disable to get all the 63k files.

			paragraphs = prepare_data(file)
			if paragraphs == None: continue

			preprocessed_data["file_ID"] = f"{file.parent.name}_{file.stem}"
			preprocessed_data["judgment"] = "\n\n".join(paragraphs)
			preprocessed_data["outcome"] = outcome

			final_data.append(preprocessed_data)

	output_path = DATASET / "SCIJuDAC.jsonl"
	output_path.parent.mkdir(exist_ok=True)

	with open(output_path, 'w') as out_file:
		for judgment in final_data: out_file.write(json.dumps(judgment) + "\n")


if __name__ == "__main__":
	main()