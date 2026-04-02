"""
Segments paragraph-wise the text from raw judgment text 
and saves them in a csv file.
"""
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

import re
import pandas as pd
from tqdm import tqdm

from config import TEXT_DATA, PARAWISE_DATA

def segmenter(filepath:Path) -> list[str]:
	"""
	Segments text paragraph-wise.
	
	Arguments
	---------
	filepath : Path
		Filepath for the judgment text to be segmented.

	Returns
	-------
	segmented_text : list[str]
		List of paragraphs.
	"""
	with open(filepath, 'r', encoding='utf-8') as f:
		try: text = f.read()
		except UnicodeDecodeError: print(filepath)

	paragraphs = re.split(r"\n\n+", text)
	segmented_text = []

	for paragraph in paragraphs:
		try: assert paragraph.strip() != ""
		except AssertionError: continue
		segmented_text.append(paragraph.strip())

	return segmented_text


def main() -> None:
	judgment_root = TEXT_DATA
	judgment_folders = sorted([folder for folder in judgment_root.iterdir() if folder.is_dir()])
	output_root = PARAWISE_DATA

	for folder in tqdm(judgment_folders, position=0, desc="Processing data"):
		# if folder.name != "1950": continue

		files = sorted([file for file in folder.iterdir() if file.is_file()])
		output_folder = output_root / folder.name
		output_folder.mkdir(exist_ok=True)

		for judgment in tqdm(files, position=1, leave=False, desc=f"{folder.name}"):
			# if judgment.name != "1.txt": continue
			segmented_judgment = segmenter(judgment)
			doc_name = judgment.stem + ".csv"
			output_filepath = output_folder / doc_name

			segmented_judgment = pd.DataFrame(segmented_judgment, columns=["text"])
			segmented_judgment.to_csv(output_filepath, index=False)


if __name__ == "__main__":
	main()