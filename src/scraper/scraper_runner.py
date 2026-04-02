"""
A script for scraping Supreme Court of India judgments
from CourtKutchehry.com. Saves the HTML files for each
judgment yearwise and extracts judgment text from them.
Also offers choices for selecting the operation
(scraping HTML+extracting data, extracting data
from already scraped HTML pages).
"""
import os, sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

from config import HTML_DATA, TEXT_DATA
from judgment_scraper import result_traversal
from judgment_extraction import judgment_extraction

START_YEAR = 1950
END_YEAR = 2025


def get_task_choice() -> int:
	"""
	Gets the task choice from the user. 1 starts the scraping,
	and 2 extracts judgment text from previously scraped HTML pages.

	Arguments
	---------
	None

	Returns
	-------
	task_choice : int
		Index of the task chosen by the user.
	"""
	options = """What do you need to do?
	1. Scraping HTML pages and extracting judgments.
	2. Extracting data from scraped HTML pages.
	Enter 1 or 2 to choose the corresponding action: 
	"""
	options = "".join(options.split("\t"))

	while True:
		try:
			task_choice = int(input(options))

			if task_choice in [1, 2]: # Runs until the value of choice becomes 1 or 2.
				return task_choice

			else:
				os.system('cls' if os.name == 'nt' else 'clear')
				task_choice = int(input("Something other than 1 or 2 entered. Would you kindly enter 1 or 2: "))

		except ValueError:
			os.system('cls' if os.name == 'nt' else 'clear')
			print("String entered. Try entering 1 or 2 instead.")


def get_year_choice() -> int:
	"""
	Asks the user for the year for which they want to save the data.
	Verifies whether the inputted year falls between the current range
	of years the data is available for or not. Also allows for working
	for all years (denoted arbitrarily by 420).

	Arguments
	---------
	None

	Returns
	-------
	year_choice : int
		The year chosen by the user., or 420 if the user wants to save for
		all the years.
	"""
	year_choice_prompt = f"""Enter a year between {START_YEAR} and {END_YEAR} 
	for which you want to save the judgment HTML page and/or text.
	Enter \"420\" for extracting from all the years: """
	year_choice_prompt = " ".join(year_choice_prompt.split()) + " "
	year_choice = int(input(year_choice_prompt))

	while (year_choice not in range(START_YEAR, END_YEAR+1)) and (year_choice != 420):
		try:
			print("Neither of the required values entered. Kindly try again.")
			year_choice = int(input(year_choice_prompt))

		except ValueError:
			print(f"String entered. Kindly enter a year between {START_YEAR} and {END_YEAR} or 420.")

	return year_choice


def get_overwrite_choice() -> str:
	"""
	Asks the user whether they want to overwrite the previously 
	written file on not.

	Arguments
	---------
	None

	Returns
	-------
	overwrite_choice : int
		The year chosen by the user., or 420 if the user wants to save for
		all the years.
	"""
	overwrite_choice = input("Do you want the program to overwrite the existing files (Y/N): ").strip().lower()

	while overwrite_choice not in ["y", "n"]:
		print("Something other than \"Y\" or \"N\" entered. Kindly enter a correct response.")
		overwrite_choice = input("Do you want the program to overwrite the existing files (Y/N): ").strip().lower()

	return overwrite_choice


def create_search_url(year:int) -> str:
	"""
	Creates the URL of the search page.

	Arguments
	---------
	year : int
		The year for which the search was done.

	Returns
	-------
	search_url : str
		Search URL for the given year.
	"""
	from_date = f"{year}-01-01"
	to_date = f"{year}-12-31"
	search_url = f"""https://www.courtkutchehry.com/advanced-search/?
	category=judgements&court_type=select&selected_courts=53
	&start_date={from_date}&end_date={to_date}
	&search_type=all&q=&at=&acts=&citation=&party_1=&party_2=&judge=&advocate=
	&bareacts_search_type=all&bareacts_q=&bareacts_at=&page=1"""
	search_url = "".join(search_url.split())

	return search_url


def scrape_and_extract_all(html_root:Path, judgment_root:Path, flag:str) -> None:
	"""
	Called when all the HTML and/or judgments are to be extracted.
	Operation depends on the value of `flag`.

	Arguments
	---------
	html_root : Path
		Root folder path for saving the scraped HTML pages.

	judgment_root : Path
		Root folder path for saving the extracted judgment text.

	flag : str
		Denotes what operation (scraping+extraction or extraction) is to be done.

	Returns
	-------
	None
	"""
	if flag == "h":
		for year in range(START_YEAR, END_YEAR+1):
			print(f"Scraping for HTML for year {year} ...")
			html_output_folder = html_root / str(year)
			judgment_output_folder = judgment_root / str(year)
			search_url = create_search_url(year)

			print("Initializing scraping and judgment extraction.")
			result_traversal(html_output_folder, search_url)
			judgment_extraction(html_output_folder, judgment_output_folder)

	elif flag == "j":
		for year in range(START_YEAR, END_YEAR+1):
			print(f"Extracting for year {year} ...")
			html_output_folder = html_root / str(year)
			judgment_output_folder = judgment_root / str(year)

			print("Initializing judgment extraction.")
			judgment_extraction(html_output_folder, judgment_output_folder)


def initiate_task(html_root:Path, judgment_root:Path, task_choice:int, year_choice:int) -> None:
	"""
	Extracts HTML+judgments or judgments based on the user's choice.

	Arguments
	---------
	html_root : Path
		Root folder path for saving the scraped HTML pages.

	judgment_root : Path
		Root folder path for saving the extracted judgment text.

	choice : int
		Index of choice for the task to be done.

	year_choice : int
		The year for which the data is to be scraped/extracted.
		If the value is 420, it scrapes/extracts for the whole range
		between `START_YEAR` and `END_YEAR`.

	Returns
	-------
	None
	"""
	if task_choice == 1: # Scraping + Text extraction
		if year_choice in range(START_YEAR, END_YEAR+1):
			html_output_folder = html_root / str(year_choice)
			judgment_output_folder = judgment_root / str(year_choice)
			search_url = create_search_url(year_choice)

			result_traversal(html_output_folder, search_url)
			judgment_extraction(html_output_folder, judgment_output_folder)

		elif year_choice == 420:
			op_flag = "h" # "h" denotes HTML+judgments. Used in `scrape_and_extract_all()`.
			scrape_and_extract_all(html_root, judgment_root, op_flag)


	elif task_choice == 2: # Text extraction only
		overwrite_flag = get_overwrite_choice()

		if year_choice in range(START_YEAR, END_YEAR+1):
			html_input_folder = html_root / str(year_choice)
			judgment_output_folder = judgment_root / str(year_choice)
			judgment_extraction(html_input_folder, judgment_output_folder, overwrite_flag)

		elif year_choice == 420:
			op_flag = "j" # "j" denotes judgments only. Used in `scrape_and_extract_all()`.
			scrape_and_extract_all(html_root, judgment_root, op_flag)


def main() -> None:
	html_output_root = HTML_DATA
	judgment_output_root = TEXT_DATA

	task_choice = get_task_choice()
	year_choice = get_year_choice()
	initiate_task(html_output_root, judgment_output_root, task_choice, year_choice)


if __name__ == "__main__":
	main()
