import re
from math import inf
from pathlib import Path

from utilities import appellate_outcome_regex


def detect_verdict_text(sentences:list[str]) -> list[str]:
	"""
	Detects whether a verdict exists within the given sentences.
	Detection is done via regex patterns defined in `regex_maker()`.

	Arguments
	---------
	sentences : list[str]
		List of preprocessed judgment sentences.

	Returns
	-------
	verdict_sentences : list[str]
		List of all the sentences in which a verdict was discovered.
	"""
	verdict_regex = appellate_outcome_regex()
	verdict_sentences = []
	for sentence in sentences:
		search = re.findall(verdict_regex, sentence)
		if search != []: verdict_sentences.append(sentence)

	return verdict_sentences


def find_verdict_sentence_sequence(search_indices_per_word:list[list[int]]) -> list[int]:
	"""
	Identifies a strictly increasing sequence of character indices--one per word--
    by treating the list of indices as a directed acyclic graph. This is used to
	find the verdict sentence in the original text.

	Arguments
	---------
	search_indices_per_word : list[list[int]]
		List of lists of indices of occurences of the searched word.

	Returns
	-------
	sequence : list[int]
		A list of character indices, one per word, representing the best-fit
        monotonic alignment. Returns an empty list if no complete alignment is
        possible.
	"""
	n_words = len(search_indices_per_word)

	# dist[i][j] = shortest distance to node (i, j)
	# prev[i][j] = predecessor index k in layer i-1

	dist = []
	prev = []
	dist.append([0] * len(search_indices_per_word[0]))
	prev.append([None] * len(search_indices_per_word[0]))

	# i = current word's index, j = current word's occurence index, k = previous word's occurence index
	for i in range(1, n_words):
		curr_len = len(search_indices_per_word[i])
		prev_len = len(search_indices_per_word[i - 1])

		dist.append([inf] * curr_len)
		prev.append([None] * curr_len)

		for j in range(curr_len):
			curr_idx = search_indices_per_word[i][j]

			for k in range(prev_len):
				prev_idx = search_indices_per_word[i - 1][k]
				if curr_idx <= prev_idx:
					continue

				new_dist = dist[i - 1][k] + (curr_idx - prev_idx)
				if new_dist < dist[i][j]:
					dist[i][j] = new_dist
					prev[i][j] = k

	last_layer = dist[-1]
	j = min(range(len(last_layer)), key=lambda x: last_layer[x])
	if last_layer[j] == inf: return []

	sequence = []
	for i in reversed(range(n_words)):
		sequence.append(search_indices_per_word[i][j])
		j = prev[i][j]
		if j is None and i != 0: return []

	return sequence[::-1]


def remove_verdict(file:Path, paragraph:str, verdicts:list[str]) -> str:
	"""
	Removes the sentences containig verdict.

	Arguments
	---------
	file : Path
		Current file for which the processing is being done.

	paragraph : str
		Input judgment paragraph from which the verdict is to be removed.

	verdicts : list[str]
		Preprocessed partial sentences of verdicts.

	Returns
	-------
	paragraph : str
		Judgment paragraph with verdict removed.
	"""
	sentences_to_be_replaced = []

	for result in verdicts:
		words = result.split(" ")
		all_indices = []
		for word in words:
			word = re.escape(word)
			word_search = list(re.finditer(word, paragraph))
			word_indices = [match.start() for match in word_search]
			all_indices.append(word_indices)

		sentence_sequence = find_verdict_sentence_sequence(all_indices)
		if sentence_sequence == []:
			print(f"Error in {file.parent.name}/{file.name}. Skipping...")
			return ""

		left = sentence_sequence[0]
		right = sentence_sequence[-1]
		punctuations = ["!", ".", "?", "\'", "\""]

		while True:
			if left == 0: break
			if paragraph[left] in punctuations: break
			left -= 1

		while True:
			if right == len(paragraph): break
			if paragraph[right] in punctuations: break
			right += 1

		sentence = re.escape(paragraph[left+1:right])
		sentences_to_be_replaced.append(sentence)

	for sentence in sentences_to_be_replaced[::-1]:
		paragraph = re.sub(sentence, "", paragraph)

	return str(paragraph)