import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

import argparse, logging, time

from config import DATASET, LOGS_ROOT
from BERT import BERT
from qwen import Qwen
from llama import Llama
from mistral import Mistral

def initiate_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, required=True, help="A name for different runs.")
	parser.add_argument('--model', type=str, default='bert', choices=["bert", "legalbert", "inlegalbert", "llama", "qwen", "mistral", "saul"], help="Model to be used.")
	parser.add_argument('--dataset', type=str, default="SCIJuDAC", choices=["ILDC_single", "ILDC_multi", "SCIJuDAC"], help="Dataset to be used")
	parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
	parser.add_argument('--epochs', type=int, default=5, help="Maximum number of epochs to train for.")
	parser.add_argument('--seed', type=int, default=42, help="Seed for training.")
	parser.add_argument('--all_toks', type=int, default=0, choices=[0, 1], help="Whether you want to use all the tokens or only the last 512 tokens.")
	parser.add_argument('--collapse_labels', type=int, default=0, choices=[0, 1], help="Whether to collapse `allowed` and `partly allowed` labels into a single `allowed`.")
	args = parser.parse_args()

	return args


def create_logger(args):
	current_date = time.strftime("%d%m%Y", time.localtime())
	logger = logging.getLogger("LJP_exp")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	log_folders = {
		"ILDC_single": "ILDC",
		"ILDC_multi": "ILDC",
		"SCIJuDAC": "SCIJuDAC",
		}
	if args.collapse_labels == 1:
		log_filepath = LOGS_ROOT / "ablation" / f"{log_folders[args.dataset]}" / f"{args.name}_{args.model}_{current_date}.log"
	else:
		log_filepath = LOGS_ROOT / f"{log_folders[args.dataset]}" / f"{args.name}_{args.model}_{current_date}.log"

	file_handler = logging.FileHandler(f"{log_filepath}")
	file_handler.setLevel(logging.INFO)
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.ERROR)
	file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	return logger


def main():
	args = initiate_parser()
	logger = create_logger(args)

	model_choices = {
		"bert":        {"cls": BERT,    "model_name": "google-bert/bert-base-uncased"},
		"legalbert":   {"cls": BERT,    "model_name": "nlpaueb/legal-bert-base-uncased"},
		"inlegalbert": {"cls": BERT,    "model_name": "law-ai/InLegalBERT"},
		"llama"      : {"cls": Llama,   "model_name": "meta-llama/Llama-3.1-8B"},
		"mistral"    : {"cls": Mistral, "model_name": "mistralai/Mistral-7B-v0.3"},
		"qwen"       : {"cls": Qwen,    "model_name": "Qwen/Qwen2.5-7B"},
		"saul"       : {"cls": Mistral, "model_name": "Equall/Saul-7B-Base"},
	}

	model_class = model_choices[args.model]["cls"]
	model_name = model_choices[args.model]["model_name"]

	dataset_choices = {
		"ILDC_single": DATASET / "ILDC_single.jsonl",
		"ILDC_multi":  DATASET / "ILDC_multi.jsonl",
		"SCIJuDAC":    DATASET / "SCIJuDAC.jsonl",
	}

	dataset_path = dataset_choices[args.dataset]

	if args.dataset in ["ILDC_single", "ILDC_multi"]:
		num_labels = 2
	else: num_labels = 3

	# Initializing an instance of the model to be trained
	model = model_class(
		dataset_path=dataset_path,
		num_labels=num_labels,
		model_name=model_name,
		batch_size=args.batch_size,
		epochs=args.epochs,
		seed=args.seed,
		all_toks=args.all_toks,
		collapse_labels=args.collapse_labels
	)

	logger.info("Training parameters:")
	arguments = vars(args)
	for arg in arguments.keys():
		logger.info(f"{arg}: {arguments[arg]}")

	logger.info("\n\n")
	print(f"Running {args.model} model for {args.dataset} dataset.")
	logger.info(f"Running {args.model} model for {args.dataset} dataset.")
	model.run()


if __name__ == "__main__":
	main()