from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re, string
import pandas as pd

from utilities import appellate_outcome_regex

stop_words = set(stopwords.words("english")) - set(["i", "my", "of"])


def preprocess_text(judgment:pd.DataFrame) -> list[str]:
	"""
	Standard preprocessing pipeline
	(lowercasing, punctuation and abbreviation standardization, stopword removal)

	Arguments
	---------
	judgment : pd.DataFrame
		Dataframe containing the judgment text.

	cleaned_sentences : list[str]
		List of final cleaned sentences containing the judgment.

	Returns
	-------
	cleaned_sentences : list[str]
		List of sentences after tokenization.
	"""
	judgment = judgment.iloc[:-1]
	separator_idx = -4 if len(judgment) > 12 else -3
	final_paragraphs = judgment["text"].iloc[separator_idx:].map(lambda x: x.lower() if isinstance(x, str) else x)
	cleaned_sentences = []

	for paragraph in final_paragraphs[::-1]:
		paragraph = re.sub(r"``|''|“|”|‘|’", "\"", paragraph)
		paragraph = re.sub(r"\bapp[\.]+", "application", paragraph)
		paragraph = re.sub(r"pet[\.]+", "petition", paragraph)
		paragraph = re.sub(r"no\.", "number", paragraph)
		sentences = sent_tokenize(paragraph)
		sentences.reverse() # To align the search to be started from the last sentence
		tokenized_sentences = [word_tokenize(sentence) for sentence in sentences[:6] if len(sentence) > 2] # Exclude 2-word sentences
		for tokenized_sentence in tokenized_sentences:
			sentence = " ".join([token for token in tokenized_sentence if token not in string.punctuation and token not in stop_words])
			cleaned_sentences.append(sentence)

	return cleaned_sentences


# def appellate_outcome_regex() -> re.Pattern:
# 	"""
# 	Creates the regular expression for finding
# 	the sentence containing the verdicts.

# 	Arguments
# 	---------
# 	None

# 	Returns
# 	-------
# 	verdict_regex : re.Pattern
# 		Regex pattern for finding the verdict sentence.
# 	"""
# 	appeal_pattern = r"(?:\s{0,1}appeal(?:s)?\b|\s{0,1}pet(?:ition)?(?:s)?\b|\s{0,1}app(?:lication)?(?:s)?\b)"

# 	numbers_pattern = r"\s(?:\d{1,4}(?:\s\d{1,4}){0,3})"
# 	multi_appeal_pattern = fr"""(?:appeal\snumber(?:s)?{numbers_pattern}|appeal(?:s)?{numbers_pattern}
# 								|pet(?:ition)?\snumber(?:s)?{numbers_pattern}|pet(?:ition)?(?:s)?{numbers_pattern}
# 								|app(?:lication)?\snumber(?:s)?{numbers_pattern}|app(?:lication)?(?:s)?{numbers_pattern})"""

# 	intervening_pattern = r"(?:\s\w+){0,3}\s"
# 	set_aside_pattern = fr"""(?:(set\saside){intervening_pattern}(?:order(?:s)?|decree(?:s)?)|
# 							(?:(?:order(?:s)?|decree(?:s)?){intervening_pattern}(set\saside)))"""

# 	allowed_pattern = r"allow(?:ed)?|succeed(?:s)?"
# 	dismissed_pattern = r"dismiss(?:ed)?|fail(?:s)?|reject(?:ed)?"
# 	decision_pattern = fr"({allowed_pattern}|{dismissed_pattern})"

# 	appeal_before_decision_pattern = fr"{appeal_pattern}{intervening_pattern}{decision_pattern}"
# 	decision_before_appeal_pattern = fr"{decision_pattern}{intervening_pattern}{appeal_pattern}"
# 	multi_appeals_decision_pattern = fr"{multi_appeal_pattern}{intervening_pattern}{decision_pattern}"

# 	verdict_pattern = appeal_before_decision_pattern + "|" + decision_before_appeal_pattern + "|" + multi_appeals_decision_pattern + "|" + set_aside_pattern
# 	verdict_regex = re.compile(verdict_pattern, re.VERBOSE)

# 	return verdict_regex


def label_extraction_from_text(judgment:pd.DataFrame) -> str:
	"""
	Extracts final judgment from last few sentences	of the judgment text.

	Arguments
	---------
	judgment : DataFrame
		Judgment text from which data is to be extracted.

	Returns
	-------
	label : str
		The outcome (allowed/dismissed/partly allowed) for the given case.
	"""
	cleaned_sentences = preprocess_text(judgment)
	verdict_regex = appellate_outcome_regex()
	opinion_regex = re.compile(r"(i\s*would\s*\b|my\s*opinion\b|of\s*opinion\b|should\b|i\s*agree\b)")

	allowed = ["allow", "allowed", "succeed", "succeeds"]
	dismissed = ["dismiss", "dismissed", "fail", "fails", "reject", "rejected"]
	verdict_counter = {"allowed":0, "dismissed":0}
	label = ""

	for sentence in cleaned_sentences:
		if len(sentence) == 1: continue
		verdict_search = re.findall(verdict_regex, sentence)
		opinion_search = re.findall(opinion_regex, sentence)

		if verdict_search and not opinion_search:
			for result in verdict_search:
				appeal_before_verdict, verdict_before_appeal, multi_appeals, set_aside_before_order, set_aside_after_order = result

				if appeal_before_verdict.strip():
					verdict = appeal_before_verdict.strip()
					if verdict in allowed: verdict_counter["allowed"] += 1
					if verdict in dismissed: verdict_counter["dismissed"] += 1

				if verdict_before_appeal.strip():
					verdict = verdict_before_appeal.strip()
					if verdict in allowed: verdict_counter["allowed"] += 1
					if verdict in dismissed: verdict_counter["dismissed"] += 1

				if multi_appeals.strip():
					verdict = multi_appeals.strip()
					if verdict in allowed: verdict_counter["allowed"] += 1
					if verdict in dismissed: verdict_counter["dismissed"] += 1

				if set_aside_before_order.strip() or set_aside_after_order.strip():
					verdict_counter["allowed"] += 1

		if (verdict_counter["allowed"] == 2 and verdict_counter["dismissed"] == 0) or (verdict_counter["allowed"] == 0 and verdict_counter["dismissed"] == 2):
			return label

		if verdict_counter["allowed"] and verdict_counter["dismissed"]: label = "partly allowed"
		elif verdict_counter["allowed"]: label = "allowed"
		elif verdict_counter["dismissed"]: label = "dismissed"

	return label


def label_extraction_from_outcome(text:str) -> str:
	"""
	Extracts the outcome label (allowed, partly allowed, disposed of, dismissed, inapplicable) 
	from the judgment text's final line. Corrects some misspells in the data, and groups them 
	within one of the four labels.

	Arguments
	---------
	text : str
		Judgment text.

	Returns
	-------
	label : str
		The outcome label (allowed/dismissed/disposed of/partly allowed/inapplicable) of the judgment.
	"""
	final_outcome_sentence = re.search(r"Final Decision\s*:\s*(.+)$", text)
	label = "inapplicable" if final_outcome_sentence == None or "" else final_outcome_sentence.group(1).lower().strip()

	if label in ["alowed", "allaowed", "alowed", "allowed/allowed", "allowed/allowed/allowed. respectively"]:
		label = "allowed"

	elif label in ["dismissing", "dismisssed", "disposed off/ dismissed, respectively", "dismissed/ disposed of"]:
		label = "dismissed"

	elif label in ["dispoded off", "dispossed off", "disposed off", "dispsoed of", "disposed",
				  "disposed  of", "disposed of.", "dispose of", "dsposed of", "disposed of/dismissed"]:
		label = "disposed of"

	elif label in [
		"allowed/dismissed", "dismissed/allowed", "partly allowed/dismissed", "allowed/dismissed, respectively", "allowed/dismissed. respectively",
		"alloweddismissed", "allowed/dismissed/allowed", "allowed/allowed/dismissed", "allowed /dismissed", "dismissed/allowed/dismissed",
		"allowed/partly allowed/dismissed, respectively", "allowed/dismissed/dismissed/dismissed /partly allowed. respectively",
		"partly allowed/ dismissed/dismissed", "dismissed/partly allowed/dismissed", "allowed/dismissed/dismissedallowed", "allowed/disposed of/dismissed",
		"allowed/dismissed/allowed/allowed/allowed/allowed/dismissed/dismissed/dismissed", "dismissed/allowed/allowed", "partly allowed.", "patly allowed",
		"allowed/allowed/dismissed/dismissed", "dismissed/dismissed/allowed", "partly allowed/allowed/allowed/dismissed", "partly", "dismissed/disposed of",
		"allowed/disposed of", "disposed of/allowed", "allowed/disposed off", "allowed/ disposed of", "allowed/disposed off", "alloweddisposed of",
		"allowed/ disposed of", "partly allowed/disposed of", "disposed of/disposed of/disposed of/disposed of/disposed of/allowed", "disposed ofdismissed"
	]:
		label = "partly allowed"

	elif label in ["overruled", "refer larger bench", "division bench refer larger bench", "division bench", "refer full bench", "refer  constitution bench"]:
		label = "inapplicable"

	return label