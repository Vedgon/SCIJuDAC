import pandas as pd
from pathlib import Path
from selenium.webdriver import Chrome # Replace with other browser if needed
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from tqdm import tqdm
import re, time, random, sys

sys.path.append(str(Path(__file__).resolve().parents[1])) # To import `config.py`
from config.config import PREPROCESSING, TEXT_DATA

START_YEAR = 1950
END_YEAR = 2025
SLEEP_TIME = 3

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


def update_url(url:str, page_num:int) -> str:
	"""
	Update or insert the "page" query parameter in a URL.

	This function parses the given URL, modifies its query string to set
	the "page" parameter to the provided page number, and reconstructs
	the updated URL. Any existing query parameters are preserved, and
	an existing "page" parameter is overwritten.

	Arguments
	---------
	url : str
		The original search page URL.

	page_num : int
		The page number to set in the URL.

	Returns
	-------
	str
		A new URL with the "page" query parameter set to `page_num`.
	"""
	parsed = urlparse(url)
	qs = parse_qs(parsed.query)
	qs["page"] = [str(page_num)]
	new_query = urlencode(qs, doseq=True)

	return urlunparse(parsed._replace(query=new_query))


def save_codes_and_titles():
	codes_n_titles = []

	for year in range(START_YEAR, END_YEAR+1):
		search_url = create_search_url(year)

		options = Options()
		options.page_load_strategy = "none"
		driver = Chrome(options=options)
		driver.get(search_url)
		WebDriverWait(driver, 60).until(
			EC.presence_of_element_located((By.CLASS_NAME, "advanced-search-results"))
		)
		driver.execute_script("window.stop();")
		driver.maximize_window()

		try:
			result_count_div = driver.find_element(By.CLASS_NAME, 'pager-hint')
			result_page_count_text = re.search(r'Page\s*(\d+)\s*of\s*(\d+)', result_count_div.text)
			result_page_count = int(result_page_count_text.group(2))

			for result_page in tqdm(range(2, result_page_count+2), desc=f"Code scraping for {year}"):
				search_results = driver.find_elements(By.CLASS_NAME, 'result-card')
				result_links = [result.find_element(By.TAG_NAME, 'a') for result in search_results]
				for result in result_links:
					codes_n_titles.append(
						{
							"code": result.get_attribute('href').split("/")[4],
							"title": result.text
						}
					)
				search_url = update_url(search_url, result_page)
				driver.get(search_url)
				time.sleep(random.randint(0, SLEEP_TIME))

		except Exception as e:
			interruption_time = time.strftime("%H:%M:%S", time.localtime())
			print("Execution rudely interrupted at", interruption_time, ".")
			print("Error:", e)

		finally:
			driver.quit()

	codes_and_titles = pd.DataFrame(codes_n_titles)
	codes_and_titles.to_csv(index=False)

	return codes_and_titles


def check_for_code(data:pd.DataFrame, codes_and_titles:pd.DataFrame):
	# Groups : 0:(lookahead), 1:(code), 2:(end of paragraph | end of sentence | mid-sentence)
	pattern = re.compile(r"(?<=\n)(\d{5,7})(\s+\d{2,3}\.|\s+\.|\s+)")
	code_pattern = re.compile(r"\d{5,7}")

	for text in data.iterrows(index=False):
		isMatch = re.search(code_pattern, text)
		if isMatch:
			matches = re.finditer(pattern, text)
			codes_and_positions = [(match.group(1).strip(), match.start(), match.end(), match.group(2).strip()) for match in matches]
			codes_and_positions.reverse()

			for s in codes_and_positions:
				continue


def main():
	codes_and_titles_filepath = PREPROCESSING / "codes_and_titles.csv"
	if codes_and_titles_filepath.exists(): codes_and_titles = pd.read_csv(codes_and_titles)
	else: codes_and_titles = save_codes_and_titles()

	# for year in range(START_YEAR, END_YEAR+1):
	# 	data_folder_path = TEXT_DATA / str(year)
	# 	files = sorted(file for file in data_folder_path.iterdir() if file.is_file())

	# 	for file in files:
	# 		data = pd.read_csv(file)
	# 		check_for_code(data, codes_and_titles)

if __name__ == '__main__':
	main()