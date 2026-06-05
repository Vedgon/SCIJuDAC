"""
Driver module to initiate the training.
Loads the model and the dataset chosen, and initializes the training.
"""
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(src_path))

import argparse, logging, time

from config import DATASET, LOGS_ROOT, COLLAPSED_LABELS_LOGS
from BERT import BicephalousBERT
from llama import BicephalousLlama
from qwen import BicephalousQwen
from mistral import BicephalousMistral


def initiate_parser() -> argparse.Namespace:
	"""
	Constructs and configures the CLI argument parser.

	Defines all CLI arguments required for training, evaluation,
    and experiment configuration.

	Arguments
	---------
	None

	Returns
	-------
	args : argparse.Namespace
		Namespace object.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, required=True, help="A name for different runs.")
	parser.add_argument('--model', type=str, default="bert", choices=["bert", "legalbert", "inlegalbert", "llama", "qwen", "mistral", "saul"], help="Model to be used.")
	parser.add_argument('--dataset', type=str, default="SCIJuDAC", choices=["ILDC_single", "ILDC_multi", "SCIJuDAC"], help="Dataset to be used")
	parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
	parser.add_argument('--epochs', type=int, default=3, help="Maximum number of epochs to train for.")
	parser.add_argument('--seed', type=int, default=42, help="Seed for training.")
	parser.add_argument('--collapse_labels', type=str, default="n", choices=["y", "n"], help="Whether to collapse `allowed` and `partly allowed` labels into a single `allowed`.")
	parser.add_argument('--load_ckpt', type=str, default="n", choices=["y", "n"], help="Whether to load checkpoint or not.")
	args = parser.parse_args()

	return args


def create_logger(args):
	"""
	Creates the experiment logger object to log the training process.
	Picks different log folders as roots based on different parser arguments.
	Log filename format: `name_modelName_datasetName_currentDate.log`
	Folders: If args.dataset == `ILDC`	   -> `ILDC`
			 If args.dataset == `SCIJuDAC` -> `SCIJuDAC`
			 If args.collapse_labels == `y` -> `collapsed_labels`

	Arguments
	---------
	args : argparse.Namespace
		The CLI-passed arguments.

	Returns
	-------
	logger : logging.Logger
		Configured logger instance.
	"""
	current_date = time.strftime("%d%m%Y", time.localtime())
	logger = logging.getLogger("LJP_exp")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	log_folders = {"ILDC_single": "ILDC", "ILDC_multi": "ILDC", "SCIJuDAC": "SCIJuDAC"}
	log_folder = f"{log_folders[args.dataset]}"
	filename = f"{args.name}_{args.model}_{args.dataset}_{current_date}.log"
	if args.collapse_labels == "y": log_filepath = COLLAPSED_LABELS_LOGS / filename
	else: log_filepath = LOGS_ROOT / log_folder / filename
	log_filepath.parent.mkdir(parents=True, exist_ok=True)

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
	"""
	Initializes training by instantiating the model and
	loading the dataset corresponding to the passed argument,
	followed by the execution of the training method.
	"""
	args = initiate_parser()
	logger = create_logger(args)

	model_choices = {
		"bert":        {"cls": BicephalousBERT,    "model_name": "google-bert/bert-base-uncased"},
		"legalbert":   {"cls": BicephalousBERT,    "model_name": "nlpaueb/legal-bert-base-uncased"},
		"inlegalbert": {"cls": BicephalousBERT,    "model_name": "law-ai/InLegalBERT"},
		"llama"      : {"cls": BicephalousLlama,   "model_name": "meta-llama/Llama-3.1-8B"},
		"qwen"       : {"cls": BicephalousQwen,    "model_name": "Qwen/Qwen2.5-7B"},
		"mistral"    : {"cls": BicephalousMistral, "model_name": "mistralai/Mistral-7B-v0.3"},
		"saul"       : {"cls": BicephalousMistral, "model_name": "Equall/Saul-7B-Base"},
	}

	model_class = model_choices[args.model]["cls"]
	model_name = model_choices[args.model]["model_name"]

	dataset_choices = {
		"ILDC_single": DATASET / "ILDC_single.jsonl",
		"ILDC_multi":  DATASET / "ILDC_multi.jsonl",
		"SCIJuDAC":    DATASET / "SCIJuDAC.jsonl",
	}

	dataset_path = dataset_choices[args.dataset]

	if args.dataset in ["ILDC_single", "ILDC_multi"] or args.collapse_labels == "y": num_labels = 2
	else: num_labels = 3

	model = model_class(
		model_name=model_name,
		dataset_path=dataset_path,
		num_labels=num_labels,
		batch_size=args.batch_size,
		epochs=args.epochs,
		seed=args.seed,
		collapse_labels=args.collapse_labels,
		load_ckpt=args.load_ckpt
	)

	logger.info("Training parameters:")
	arguments = vars(args)
	for arg in arguments.keys():
		logger.info(f"{arg}: {arguments[arg]}")

	print(f"Running {args.model} model for {args.dataset} dataset.")
	logger.info(f"Running {args.model} model for {args.dataset} dataset.\n")
	model.run()


if __name__ == "__main__":
	main()