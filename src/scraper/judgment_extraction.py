from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


def judgment_writer(html_file_path:Path, judgment_file_path:Path) -> None:
	"""
	Writes the judgment data to the disk by extracting it 
	from the scraped HTML file.

	Arguments
	---------
    html_file_path : Path
	    Path to the scraped HTML file.

	judgment_file_path : Path
        Path to the txt file where the judgment text is to be stored.

	Returns
	-------
    None
	"""
	with open(html_file_path, 'r', encoding='utf-8') as h:
		soup = BeautifulSoup(h, 'html.parser')
		judgment_div = soup.find('article', {'id':'fullText'})
		case_metadata = soup.find('ul', {'class':'list-unstyled'})
		decision = case_metadata.find_all('li')[-1].text
		decision = decision.replace("Result: ", "")
		paragraph_divs = judgment_div.find_all('p')

		paragraphs = [{"text": para_div.text} for para_div in paragraph_divs if para_div.text != ""]
		paragraphs.append({"text": decision})

	judgment = pd.DataFrame(paragraphs)
	judgment.to_csv(path_or_buf=judgment_file_path, index=False)


def judgment_extraction(html_root_folder:Path, judgment_root_folder:Path, flag="y") -> None:
	"""
	Initiates the extraction of the judgment text from the saved HTML pages.
	Has the option to let the user overwrite the existing data via the `flag` variable.

	Arguments
	---------
    html_root_folder : Path
        Root folder path for the scraped HTML files.

	judgment_root_folder : Path
        Root folder path for the extracted judgment data.

	flag : str
        Overwrite flag. Can have 'y' (yes) or 'n' (no) as values.

	Returns
	-------
	None
	"""
	html_files = sorted([file for file in html_root_folder.iterdir() if file.is_file()])
	judgment_root_folder.mkdir(exist_ok=True)

	for html_file in tqdm(html_files, desc="File writing", colour="#03a5fc"):
		html_file_path = html_root_folder / html_file
		judgment_file_path = judgment_root_folder / f"{html_file.stem}.csv"

		if flag == "n" and not judgment_file_path.exists():
			judgment_writer(html_file_path, judgment_file_path)

		elif flag == "y":
			judgment_writer(html_file_path, judgment_file_path)
