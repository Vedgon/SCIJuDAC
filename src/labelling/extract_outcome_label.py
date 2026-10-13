from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re, string
import pandas as pd

from utilities import appellate_outcome_regex

stop_words = set(stopwords.words("english")) - set(["i", "my", "of"])

# A label-only exclusion guard, not a polarity inversion rule.
_OUTCOME_WORD = r"(?:allow(?:ed|ing)?|dismiss(?:ed|ing)?|reject(?:ed|ing)?|succeed(?:s)?|fail(?:s)?|set\s+aside|dispos(?:e|ed)\s+of)"
_NEG_LINK = r"(?:be|been|being|have|has|had|to|therefore|thus|accordingly|hereby|possibly|properly|legally|simply|merely|partly|partially|wholly|entirely|further|ultimately|necessarily)"
_NEGATED_OUTCOME = re.compile(
    r"\b(?:not|never|cannot|unable\s+to|decline\s+to|refuse\s+to|"
    r"no\s+(?:reason|ground|grounds|basis|justification)\s+(?:to|for))\s+"
    r"(?:" + _NEG_LINK + r"\s+){0,6}"
    r"(?P<outcomes>" + _OUTCOME_WORD +
    r"(?:\s+(?:or|nor)\s+(?:" + _NEG_LINK + r"\s+){0,3}" + _OUTCOME_WORD + r")*)\b"
)
_OUTCOME_TOKEN = re.compile(r"\b" + _OUTCOME_WORD + r"\b")


def _guard_negated_outcomes(paragraph: str) -> str:
    """Mask only outcome words in recognized negative constructions.

    Applied before stopword removal and to BOTH label matching paths. Length
    and punctuation are retained. No changes are made to the input document.
    Unrecognized syntax remains a limitation; this is not general scope parsing.
    """
    chars = list(paragraph)
    for match in _NEGATED_OUTCOME.finditer(paragraph):
        for outcome in _OUTCOME_TOKEN.finditer(paragraph, *match.span("outcomes")):
            for index in range(outcome.start(), outcome.end()):
                if not paragraph[index].isspace():
                    chars[index] = "x"
    return "".join(chars)


# Label-only additions: do not alter the shared leakage-removal regex.
_NUMBER = r"\d+(?:\s*[/–-]\s*\d+)*(?:\s+of\s+\d{4})?"
_APPEAL = (r"\b(?:civil\s+|criminal\s+)?appeals?\b"
           r"(?:\s+(?:numbers?\s*)?" + _NUMBER +
           r"(?:\s*(?:,|and|&)\s*" + _NUMBER + r")*)?")
_DECISION = r"(?P<decision>(?:partly\s+|partially\s+)?allowed|dismissed|rejected|disposed\s+of)\b"
_DIRECT = re.compile(_APPEAL + r"\s+(?:(?:is|are)\s+)?(?:hereby\s+|accordingly\s+)?" + _DECISION)
_PRONOUN = re.compile(r"^\s*it\s+is\s+(?:hereby\s+|accordingly\s+)?" + _DECISION)


def _additional_outcomes(paragraphs: list[str]) -> set[str]:
    """Narrow raw-text rules with same-paragraph, adjacent-sentence context."""
    outcomes = set()
    for paragraph in paragraphs:
        previous = ""
        for sentence in sent_tokenize(paragraph)[-6:]:
            matches = list(_DIRECT.finditer(sentence))
            pronoun = _PRONOUN.search(sentence)
            # Only resolve 'it' after one singular appeal explicitly described
            # as infructuous/non-surviving; never carry context across paragraphs.
            if (pronoun and len(re.findall(r"\bappeals?\b", previous)) == 1
                    and re.search(r"\bappeal\b", previous)
                    and re.search(r"\b(?:infructuous|does not survive)\b", previous)
                    and not re.search(r"\b(?:petition|application|suit|order|decree)\b", previous)
                    and not re.search(r"\b(?:would|should|could|if|was|were|had)\b", previous)):
                matches.append(pronoun)
            for match in matches:
                # Avoid straightforward reported, hypothetical or quoted outcomes.
                prefix = sentence[:match.start()]
                if re.search(r"\b(?:if|would|should|could|held|said|submitted|contended)\b", prefix):
                    continue
                if '"' in sentence or re.search(r"\b(?:high court|trial court)\b", prefix):
                    continue
                value = match.group("decision")
                outcomes.add("dismissed" if value == "rejected" else
                             "partly allowed" if value == "partially allowed" else value)
            previous = sentence
    return outcomes


def preprocess_text(judgment:pd.DataFrame) -> list[str]:
	"""
	Standard preprocessing pipeline
	(lowercasing, punctuation and abbreviation standardization, stopword removal)

	Arguments
	---------
	judgment : pd.DataFrame
		Dataframe containing the judgment text.

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
		paragraph = re.sub(r"\b(can|could|would|should|is|are|was|were|has|have|had|do|does|did)n['’]t\b", r"\1 not", paragraph)
		paragraph = re.sub(r"\bwon['’]t\b", "will not", paragraph)
		paragraph = re.sub(r"\bshan['’]t\b", "shall not", paragraph)
		paragraph = re.sub(r"\b(are|is|was|were)(dismissed|allowed|rejected)\b", r"\1 \2", paragraph)
		paragraph = re.sub(r"``|''|“|”|‘|’", "\"", paragraph)
		paragraph = re.sub(r"\bapp[\.]+", "application", paragraph)
		paragraph = re.sub(r"pet[\.]+", "petition", paragraph)
		paragraph = re.sub(r"nos?\.", "number", paragraph)
		sentences = sent_tokenize(paragraph)
		sentences.reverse() # To start the search from the last sentence
		tokenized_sentences = [word_tokenize(sentence) for sentence in sentences[:6] if len(sentence) > 2] # Exclude 2-word sentences
		for tokenized_sentence in tokenized_sentences:
			sentence = " ".join([token for token in tokenized_sentence if token not in string.punctuation and token not in stop_words])
			cleaned_sentences.append(sentence)

	return cleaned_sentences


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
	additional = _additional_outcomes(cleaned_sentences)
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
			if verdict_counter["allowed"]: label = "allowed"
			else: label = "dismissed"
			return label

		if verdict_counter["allowed"] and verdict_counter["dismissed"]: label = "partly allowed"
		elif verdict_counter["allowed"]: label = "allowed"
		elif verdict_counter["dismissed"]: label = "dismissed"

	return label


def label_extraction_from_ck(text:str) -> str:
	"""
	Extracts the outcome label (allowed, partly allowed, disposed of, dismissed, inapplicable) 
	from the judgment data's final row. Corrects some misspells in the data, and groups them 
	within one of the four labels.

	Arguments
	---------
	text : str
		Final row of the judgment.

	Returns
	-------
	label : str
		The outcome label (allowed/dismissed/disposed of/partly allowed/inapplicable) of the judgment.
	"""
	final_outcome_sentence = re.search(r"Final Decision\s*:\s*(.+)$", text)
	label = "inapplicable" if final_outcome_sentence == None or "" else final_outcome_sentence.group(1).lower().strip()

	if label in ["alowed", "allaowed", "alowed", "allowed/allowed", "allowed/allowed/allowed. respectively"]:
		label = "allowed"

	elif label in ["dismissing", "dismisssed", "disposed off/ dismissed, respectively", "dismissed/ disposed of", "disposed of/dismissed"]:
		label = "dismissed"

	elif label in ["dispoded off", "dispossed off", "disposed off", "dispsoed of", "disposed",
				  "disposed  of", "disposed of.", "dispose of", "dsposed of"]:
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
