"""
Fixes the  SCI judgment data obtained from CourtKutchery.com (CK).
The original data contains URL codes for judgments hosted at CK 
for judgments referred within a judgment text, instead of the actual 
title of the judgment. The following code fixes that by replacing the
code with its corresponding title.
"""
import re, csv
from pathlib import Path
from config import TEXT_DATA, BACKUP

START_YEAR = 1950
END_YEAR = 1951


def open_replacement_data(codes_and_titles_filepath:Path, missingNo_filepath:Path) -> tuple[dict[int, str], set[int]]:
	"""
	Opens the files which contain the codes and corresponding titles (constructed separately), 
	and all the potential codes (collected separately).

	Arguments
	---------
	codes_and_titles_filepath : pathlib.Path
		File path for codes_and_titles.csv.

	missingNo_filepath : pathlib.Path
		File path for missingNo.csv.

	Returns
	-------
	(codes_and_titles, missingNo) : tuple[dict[int, str], set[int]]
		Tuple of codes_and_titles (code and corresponding titles), and missingNo (potential codes).
	"""

	codes_and_titles = dict()
	missingNo = set()

	with open(codes_and_titles_filepath, 'r', encoding="utf-8") as ctf, open(missingNo_filepath, 'r', encoding="utf-8") as mnf:
		next(ctf)
		ctd = csv.reader(ctf)
		temp = [(row[0], row[1]) for row in ctd]
		for d in temp:
			codes_and_titles[d[0]] = d[1]
		
		mnd = csv.reader(mnf)
		for row in mnd:
			for item in row:
				missingNo.add(item)

	return (codes_and_titles, missingNo)


def printer(codes_and_titles:dict[int, str], codes_and_positions:list[tuple[str, int, int, str]]) -> None:
	"""
	Prints the output after replacing the code. For debugging purpose.

	Arguments
	---------
	codes_and_titles : dict[int, str]
		Contains the codes and their corresponding titles.

	codes_and_positions : list[tuple[str, int, int, str]]
		Contains the code matched via regex, starting and ending indices, and the character succeeding the matched code.

	Returns
	-------
	None
	"""

	for code, match_start, match_end, par_end in codes_and_positions:
		if code in codes_and_titles.keys():
			if par_end == "": # Mid-sentence
				judgment_text = judgment_text[:match_start-1] + f"{codes_and_titles[code]} " + judgment_text[match_end:] # match_start-1 to remove the extra whitespace
				print("Mid-sentence.")
				print("Text to be replaced:", code)
				print("Text to be added:", codes_and_titles[code])
				print(judgment_text[match_start-50:match_end+100])
				print("----------")

			elif par_end == ".": # End of sentence.
				judgment_text = judgment_text[:match_start-1] + f"{codes_and_titles[code]}." + judgment_text[match_end:]
				print("End-sentence.")
				print("Text to be replaced:", code)
				print("Text to be added:", codes_and_titles[code])
				print(judgment_text[match_start-50:match_end+100])
				print("----------")

			elif len(par_end) >= 2: # End of paragraph
				judgment_text = judgment_text[:match_start-1] + f"{codes_and_titles[code]}.\n\n{par_end}" + judgment_text[match_end:]
				print("End-of-paragraph.")
				print("Text to be replaced:", code)
				print("Text to be added:", codes_and_titles[code])
				print(judgment_text[match_start-50:match_end+100])
				print("----------")


def cleanup(judgment_text:str, codes_and_titles:dict[int, str], missingNo:set[int], matches:re.Match) -> str:
	"""
	Fixes the judgment text by replacing the code with its corresponding title.

	Argument
	--------
	judgment_text : str
		Judgment text with codes instead of proper text.

	codes_and_titles : dict[int, str]
		Contains the codes and their corresponding titles.

	missingNo : set[int]
		Contains potentially missing codes.

	matches : Iterator[Match(str)]
		Regex iterator containing all the Match objects.

	Returns
	-------
	cleaned_text : str
		Judgment text after clean-up. 
	"""

	cleaned_text = ""
	codes_and_positions = [(match.group(1).strip(), match.start(), match.end(), match.group(2).strip()) for match in matches]
	codes_and_positions.reverse()
	print(codes_and_positions)

	# printer(codes_and_titles, codes_and_positions)

	for code in missingNo:
		if code in judgment_text:
			cleaned_text = judgment_text.replace(code, codes_and_titles[code])

	return cleaned_text


def write_output(cleaned_text:str, output_root:Path, year:str, filename:str) -> None:
	"""
	Writes the cleaned text as txt file.

	Arguments
	---------
	cleaned_text : str
		Text obtained after replacing codes with corresponding case titles.

	output_root : pathlib.Path
		Root directory path for output.

	year : str
		Folder name for the output file.

	file : str
	 Filename for the output.

	Returns
	-------
	None
	"""
	if cleaned_text != []:
		output_folder_yearly = output_root / year
		if not output_folder_yearly.exists(): output_folder_yearly.mkdir()
		output_filepath = output_folder_yearly / filename
		with open(output_filepath, 'w', encoding="utf-8") as of:
			of.write(cleaned_text)

	else:
		print(f"Text non-existent. Not writing {filename} for {year}.")


def main():
	data_root = TEXT_DATA
	output_root = BACKUP
	codes_and_titles_filepath = output_root / "codes_and_titles.csv"
	missingNo_filepath = output_root / "missingNo_CD.csv"

	codes_and_titles, missingNo = open_replacement_data(codes_and_titles_filepath, missingNo_filepath)

	years = [str(year) for year in range(START_YEAR, END_YEAR)]
	pattern = re.compile(r"(?<=\n)(\d{5,7})(\s+\d{2,3}\.|\s+\.|\s+)") # Groups : 0:(lookahead), 1:(code), 2:(end of paragraph | end of sentence | mid-sentence)

	code_pattern = re.compile(r"\d{5,7}")

	for year in years:
		print(f"Searching in {year}...")
		data_yearly = data_root / str(year)
		judgment_files = sorted(file for file in data_yearly.iterdir() if file.is_file())

		for file in judgment_files:
			with open(file, 'r', encoding="utf-8") as jf:
				judgment_text = jf.read()
				matches = re.finditer(pattern, judgment_text)
				isMatch = re.search(code_pattern, judgment_text)

				if isMatch:
					print(file)
					cleaned_text = cleanup(judgment_text, codes_and_titles, missingNo, matches)

				write_output(cleaned_text, output_root, year, file)


if __name__ == '__main__':
	main()