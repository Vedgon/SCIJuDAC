from pathlib import Path
from selenium.webdriver import Chrome # Replace with other browser if needed
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from tqdm import tqdm
import codecs, re, time, random, traceback

SLEEP_TIME = 3
WAIT_TIME = 60


def save_html_pages(driver:Chrome, urls:list[str], file_id:int, html_output_folder:Path) -> None:
	"""
	Sequentially opens the judgments stored in `urls` and saves the HTML
	source file to the given path.

	Arguments
	---------
	driver : webdriver
		Chrome webdriver object.
	
	urls : list[str]
		List of URLS per search result.
	
	file_id : int
		File number. To be used as filename.
	
	html_output_root:Path
		Root folder path for scraped HTML pages.
	
	next_page_link:str
		Link of the next page of the search result.

	Returns
	-------
	None
	"""
	search_page = driver.current_window_handle
	driver.switch_to.new_window('tab')
	judgment_tab = driver.current_window_handle
	driver.switch_to.window(judgment_tab)

	for url in urls:
		html_filename = f"{file_id}.html"
		html_file_path = html_output_folder / html_filename
		driver.get(url)

		try:
			wait = WebDriverWait(driver, timeout=WAIT_TIME)
			wait.until(EC.visibility_of_element_located((By.CLASS_NAME, 'judgement-content')))
			source = driver.page_source

			with codecs.open(html_file_path, 'w', encoding='utf-8') as h:
				h.write(source)

			time.sleep(random.randint(0, SLEEP_TIME))

		except Exception as e:
			interruption_time = time.strftime("%H:%M:%S", time.localtime())
			print("Timeout error.")
			print("Code rudely timed out at", interruption_time, ".")
			print("Error:", e)
			traceback.print_exc()

		file_id += 1

	driver.close()
	driver.switch_to.window(search_page)


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


def result_traversal(html_output_folder:Path, search_url:str) -> None:
	"""
	Goes through all the results found for per year's search
	and saves the HTML file for each judgment. File names
	are given according to the result's position.

	Arguments
	---------
	html_output_root : Path
		Root folder path for the scraped HTML pages.

	search_url : str
		URL of the search results.

	Returns
	-------
	None
	"""
	html_output_folder.mkdir(parents=True, exist_ok=True)
	options = Options()
	options.page_load_strategy = "none"
	driver = Chrome(options=options)
	driver.get(search_url)
	WebDriverWait(driver, 60).until(
		EC.presence_of_element_located((By.CLASS_NAME, "advanced-search-results"))
	)
	driver.execute_script("window.stop();")
	driver.maximize_window()
	file_id = 1

	try:
		result_count_div = driver.find_element(By.CLASS_NAME, 'pager-hint')
		result_page_count_text = re.search(r'Page\s*(\d+)\s*of\s*(\d+)', result_count_div.text)
		result_page_count = int(result_page_count_text.group(2))

		for result_page in tqdm(range(2, result_page_count+2), desc=f"{html_output_folder.name}"):
			search_results = driver.find_elements(By.CLASS_NAME, 'result-card')
			result_links = [result.find_element(By.TAG_NAME, 'a').get_attribute('href') for result in search_results]
			save_html_pages(driver, result_links, file_id, html_output_folder)
			time.sleep(random.randint(0, SLEEP_TIME))
			file_id += len(result_links)
			search_url = update_url(search_url, result_page)
			driver.get(search_url)

	except Exception as e:
		interruption_time = time.strftime("%H:%M:%S", time.localtime())
		print("Execution rudely interrupted at", interruption_time, ".")
		print("Error:", e)
		traceback.print_exc()

	finally:
		driver.quit()