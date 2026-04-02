import re

def appellate_outcome_regex() -> re.Pattern:
	"""
	Creates the regular expression for finding
	the sentence containing the verdicts.

	Arguments
	---------
	None

	Returns
	-------
	verdict_regex : re.Pattern
		Regex pattern for finding the verdict sentence.
	"""
	appeal_pattern = r"(?:\s{0,1}appeal(?:s)?\b|\s{0,1}pet(?:ition)?(?:s)?\b|\s{0,1}app(?:lication)?(?:s)?\b)"

	numbers_pattern = r"\s(?:\d{1,4}(?:\s\d{1,4}){0,3})"
	multi_appeal_pattern = fr"""(?:appeal\snumber(?:s)?{numbers_pattern}|appeal(?:s)?{numbers_pattern}
								|pet(?:ition)?\snumber(?:s)?{numbers_pattern}|pet(?:ition)?(?:s)?{numbers_pattern}
								|app(?:lication)?\snumber(?:s)?{numbers_pattern}|app(?:lication)?(?:s)?{numbers_pattern})"""

	intervening_pattern = r"(?:\s\w+){0,3}\s"
	set_aside_pattern = fr"""(?:(set\saside){intervening_pattern}(?:order(?:s)?|decree(?:s)?)|
							(?:(?:order(?:s)?|decree(?:s)?){intervening_pattern}(set\saside)))"""

	allowed_pattern = r"allow(?:ed)?|succeed(?:s)?"
	dismissed_pattern = r"dismiss(?:ed)?|fail(?:s)?|reject(?:ed)?"
	decision_pattern = fr"({allowed_pattern}|{dismissed_pattern})"

	appeal_before_decision_pattern = fr"{appeal_pattern}{intervening_pattern}{decision_pattern}"
	decision_before_appeal_pattern = fr"{decision_pattern}{intervening_pattern}{appeal_pattern}"
	multi_appeals_decision_pattern = fr"{multi_appeal_pattern}{intervening_pattern}{decision_pattern}"

	verdict_pattern = appeal_before_decision_pattern + "|" + decision_before_appeal_pattern + "|" + multi_appeals_decision_pattern + "|" + set_aside_pattern
	verdict_regex = re.compile(verdict_pattern, re.VERBOSE)

	return verdict_regex