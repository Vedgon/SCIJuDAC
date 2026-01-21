"""
The code follows the methods described in the paper "ILDC for CJPE"
by Malik et. al. to create embeddings.
"""

import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np


def chunk_long_sequence(input_ids):
	chunked_input_ids = []
	chunk_size = 510
	stride = 410
	for i in range(0, len(input_ids), stride):
		chunk = input_ids[i:i+chunk_size]
		if len(chunk) > 0: chunked_input_ids.append(chunk)

	return chunked_input_ids


def get_embeddings(input_ids, attention_mask, model):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
	attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

	with torch.no_grad():
		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		last_hidden = outputs.last_hidden_state
		print(f"Last hidden layer's shape: {last_hidden.shape}")
		mask = attention_mask.unsqueeze(-1)
		vec = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1) # Shape: (1, 768)

	print(f"Embedding vector's shape: {vec.shape}, Attention mask's shape: {mask.shape}")

	return vec.squeeze(0).cpu().numpy() # Shape: (768, )


def create_embeddings(x_batch:list, y_batch:list, embedding_model:transformers.models, tokenizer:transformers, version:str):
	pretrained_model = embedding_model.from_pretrained(version)
	if version == "xlnet-base-cased":
		model = pretrained_model
		add_prefix_space = True
	else:
		model = pretrained_model
		add_prefix_space = False
	tokenizer = tokenizer.from_pretrained(version, add_prefix_space=add_prefix_space)
	embedding_vecs = []

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.eval()
	model.to(device)

	for judgment in x_batch:
		tokenized_judgment = tokenizer.tokenize(judgment)
		if len(tokenized_judgment) > 10000:
			tokenized_judgment = tokenized_judgment[len(tokenized_judgment)-10000:]
		chunked_input_ids = chunk_long_sequence(tokenized_judgment)

		encoded_chunks = []
		for chunk in chunked_input_ids:
			if version == "xlnet-base-cased":
				chunk = chunk + [tokenizer.sep_token] + [tokenizer.cls_token]
			else:
				chunk = [tokenizer.cls_token] + chunk + [tokenizer.sep_token]
			encoded_chunk = tokenizer.convert_tokens_to_ids(chunk)
			encoded_chunks.append(encoded_chunk)

		input_ids = pad_sequences(
			encoded_chunks,
			maxlen=512,
			value=0,
			dtype="long",
			truncating="pre",
			padding="pre" if version=="xlnet-base-cased" else "post"
		) # Shape: (# of chunks, 512)

		attention_masks = [[int(token_id > 0) for token_id in seq] for seq in input_ids]
		print(f"Input IDs' shape: {input_ids.shape}, Attention mask's shape: {len(attention_masks), len(attention_masks[-1])}")

		embedding_vecs_per_judgment = []
		for idx, input_id in enumerate(input_ids):
			vec = get_embeddings(input_id, attention_masks[idx], model)
			embedding_vecs_per_judgment.append(vec)

		embedding_vecs.append(np.stack(embedding_vecs_per_judgment, axis=0))
		# Shape: (# of chunks, 1, 768)
		print(f"Last embedding's shape: {embedding_vecs_per_judgment[-1].shape}, Judgment embedding's shape: {embedding_vecs[-1].shape}")

	largest_chunk_size = max(e.shape[0] for e in embedding_vecs)
	num_embeddings = len(embedding_vecs)
	print(f"Size of largest chunk: {largest_chunk_size}, Total number of embeddings created: {num_embeddings}")

	x_embds = np.full((num_embeddings, largest_chunk_size, 768), -99., dtype=np.float32)
	y_embds = np.zeros((len(y_batch), 1))

	for i, e in enumerate(embedding_vecs):
		n_i = e.shape[0] # Value: # of chunks
		print(n_i)
		x_embds[i, :n_i] = e
	
	for i, y_b in enumerate(y_batch):
		y_embds[i] = y_b

	print(f"Batch embedding's shape: {x_embds.shape}, {y_embds.shape}")
	return x_embds, y_embds


def choose_embeddings(x_batch:list, y_batch:list, choice:str):
	"""
	
	"""
	choices = {
		"bert": ["BertModel", "BertTokenizer", "bert-base-uncased"],
		"roberta": ["RobertaModel", "RobertaTokenizer", "roberta-base"],
		"xlnet": ["XLNetModel", "XLNetTokenizer", "xlnet-base-cased"]
	}

	model_name, tokenizer_name, version = choices[choice]
	embedding_model = getattr(transformers, model_name)
	tokenizer = getattr(transformers, tokenizer_name)

	embeddings = create_embeddings(x_batch, y_batch, embedding_model, tokenizer, version)

	return embeddings


def main():
	from tqdm import tqdm
	from torch.utils.data import DataLoader
	from datasets import load_dataset
	from transformers import logging
	logging.set_verbosity_error()
	import math

	dataset_path = "/scratch/anurag_y_cs.iitr/SCIJuDAC/data/SCIJuDAC.jsonl"
	dataset = load_dataset("json", data_files=str(dataset_path), split="train")
	dataset = dataset.select(range(1))
	data = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

	x, y_raw = [], []
	for batch in tqdm(data, desc="X-Y loader"):
		judgment = ""
		for paragraph in batch["judgment"]:
			judgment += paragraph["text"][0]
		x.append(judgment)
		y_raw.append(batch["outcome"][0])

	map = {"dismissed": "0", "allowed": "1", "partly allowed": "2"}
	y = [map[label] for label in y_raw]

	embedding_model_choice = "xlnet"
	x_embed, y_embed = [], []

	for batch_idx in tqdm(range(0, len(x), 32), desc="Embedding per batch", total=math.ceil(len(x)/32)):
		end_idx = min(batch_idx + 32, len(x))
		x_batch = x[batch_idx:end_idx]
		y_batch = y[batch_idx:end_idx]
		x_batch_embed, y_batch_embed = choose_embeddings(x_batch, y_batch, embedding_model_choice)
		x_embed.append(x_batch_embed)
		y_embed.append(y_batch_embed)

	np.save("/scratch/anurag_y_cs.iitr/SCIJuDAC/embeddings/XLNet_data.npy", x_embed)
	np.save("/scratch/anurag_y_cs.iitr/SCIJuDAC/embeddings/XLNet_label.npy", y_embed)

if __name__ == '__main__':
	main()
