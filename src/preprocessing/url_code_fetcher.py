import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

import time, random, csv
from selenium import webdriver
from selenium.webdriver.common.by import By

from config import PREPROCESSING

missingNo_output_path = PREPROCESSING / "missingNo.csv"
codes_and_titles_root = PREPROCESSING / "codes_and_titles"
codes = []

with open(missingNo_output_path, 'r', encoding="utf-8") as mop:
	mof = csv.reader(mop)
	for row in mof:
		for code in row:
			codes.append(code)

print("Total number of missingNos:", len(codes))
base_url = "https://www.courtkutchehry.com/Judgement/Search/t/"
codes_and_titles = {}
driver = webdriver.Chrome()

for code in codes:
	url = base_url + str(code)
	try:
		driver.get(url)
		title = driver.find_element(By.ID, "titleOfPage").get_attribute("innerText")
		if "<BR>" in title:
			title = title.replace("<BR>", "|")
		codes_and_titles[code] = title
		time.sleep(random.randint(0, 5))
		# print(codes_and_titles)

	except Exception as e:
		print(f"{code} skipped due to {e}.")
		continue

years = [str(year) for year in range(1950, 2026)]
old_codes_and_titles = {}

for year in years:
	yearly_code_and_titles_filepath = codes_and_titles_root / f"{year}.csv"
	with open(yearly_code_and_titles_filepath, 'r', encoding="utf-8") as yct:
		next(yct)
		ctf = csv.reader(yct)
		temp = [(row[0], row[1]) for row in ctf]
		for code, title in temp:
			if "<BR>" in title:
				title = title.replace("<BR>", "|")
			old_codes_and_titles[code] = title

codes_and_titles.update(old_codes_and_titles)
print(len(old_codes_and_titles), len(codes_and_titles))

output_path = codes_and_titles_root / "codes_and_titles.csv"
with open(output_path, 'w', encoding="utf-8") as op:
	out_file = csv.writer(op)
	out_file.writerow(["URL_code", "Title"])
	for code in codes_and_titles:
		out_file.writerow([code, codes_and_titles[code]])