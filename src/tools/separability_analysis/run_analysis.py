"""
Driver module to initiate analysis of models.
Loads the model and the dataset, and performs spearability and calibration analysis.
"""
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

import argparse, logging, time
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from config.config import DATASET, IMAGES_ROOT, SEPARABILITY_AND_CALIBRATION_LOGS
from loader import DataAndModelLoader
from analysis import Analysis


def initiate_parser():
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
	parser.add_argument('--model', type=str, default='bert', choices=["bert", "legalbert", "inlegalbert", "llama", "qwen", "mistral", "saul"], help="Model to be used.")
	parser.add_argument('--dataset', type=str, default="SCIJuDAC", choices=["ILDC_single", "ILDC_multi", "SCIJuDAC"], help="Dataset to be used")
	parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
	parser.add_argument('--seed', type=int, default=42, help="Seed for training.")
	parser.add_argument('--load_lc_ckpt', type=str, default="n", choices=["y", "n"], help="Load the label collapse ablation checkpoint or not.")
	parser.add_argument('--load_cc_ckpt', type=str, default="n", choices=["y", "n"], help="Load the classifier collapse ablation checkpoint or not.")
	parser.add_argument('--load_ckpt', type=str, choices=["y"], default="y", help="Whether to load full training checkpoint or not.") # Keep as-is to load the checkpoint for analysis.
	parser.add_argument('--enable_quant', type=str, choices=["y", "n"], default="y", help="Whether to quantize the model or not (should be \"n\" for BERT-based models, \"y\" for the rest).")
	args = parser.parse_args()

	return args


def create_logger(args:argparse.Namespace):
	"""
	Creates the analysis logger object to log the analysis process.
	Picks different log folders as roots based on different parser arguments.
	Log filename format: `name_modelName_datasetName_currentDate.log`.
	Prepends "ablation" if `args.collapse_labels="y"`
	Folders: If args.dataset == `ILDC`	    -> `ILDC`
			 If args.dataset == `SCIJuDAC`  -> `SCIJuDAC`
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
	logger = logging.getLogger("ablation_exp")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	filename = f"{args.name}_{args.model}_{args.dataset}_{current_date}.log"
	if args.load_lc_ckpt == "y":
		log_filepath = SEPARABILITY_AND_CALIBRATION_LOGS / "collapsed_labels" / filename
	elif args.load_cc_ckpt == "y":
		log_filepath = SEPARABILITY_AND_CALIBRATION_LOGS / "collapsed_classifier" / filename
	else:
		log_filepath = SEPARABILITY_AND_CALIBRATION_LOGS / f"{args.dataset}" / filename

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


def plot_umap(filename:str, embeddings, labels, args:argparse.Namespace):
	"""
	Plot a two-dimensional UMAP projection of model embeddings.

	The embeddings are projected into two dimensions using cosine
	distance and plotted according to their ground-truth class labels.
	The resulting figure is saved to the configured image directory.

	Parameters
	----------
	filename : str
		Base name used for the saved UMAP image.

	embeddings : Tensor or ndarray
		Model embeddings to project into two dimensions.

	labels : Tensor or ndarray
		Ground-truth class labels corresponding to the embeddings.

	args : argparse.Namespace
		Parsed command-line arguments containing the model
		configuration.

	Returns
	-------
	None
	"""
	reducer = UMAP(
		n_neighbors=15,
		min_dist=0.1,
		metric="cosine",
		random_state=42
	)
	embedding_2d = reducer.fit_transform(embeddings)
	classes = np.unique(labels)
	colors = ["#0072B2", "#D55E00", "#009E73"]
	plt.figure(figsize=(8, 6))
	for cls, color in zip(classes, colors):
		mask = labels == cls
		plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], color=color, label=f"Class {cls}", s=40)

	plt.legend(title="Class")
	plt.title(f"UMAP projection for {args.model}")
	image_filepath = IMAGES_ROOT / f"umap_{filename}.png"
	plt.savefig(image_filepath)


def log_metrics(logger:logging.Logger, results:dict, load_cc_ckpt:str) -> None:
	"""
	Logs the analysis metrics in the log file.

	Arguments
	---------
	logger : logging.Logger
		Logger instance.

	results : dict
		Dictionary containing all the metric values, which includes
		Brier score, ECE score, Silhouette score, DBI score,
		coarse logit margin scores (mean values of total, correct,
		and incorect, and all the correct and incorrect margin
		differences), and the classification report.

	load_cc_ckpt : str
		Flag indicating whether the loaded checkpoint uses the
		hierarchical or the collapsed classifier architecture.

	Returns
	-------
	None
	"""
	logger.info("Brier score: %f", results["brier_score"])
	logger.info("Expected Calibration score: %f", results["ece_score"])
	logger.info("Silhouette score: %f", results["silhouette_value"])
	logger.info("Davies-Bouldin score: %f\n\n", results["davies_bouldin_index"])

	if load_cc_ckpt == "y":
		logger.info("Logit margin scores")
		for key, score in results["logit_margin_scores"].items():
			if key in ["Correct Margins", "Incorrect Margins"]:
				logger.info(f"{key}:")
				[logger.info(f"{value}") for value in score]
			else:
				logger.info(f"{key}: {score}")

	else:
		logger.info(f"Coarse logits margin scores:")
		for key, score in results["coarse_logit_margin_scores"].items():
			if key in ["Correct Margins", "Incorrect Margins"]:
				logger.info(f"{key}:")
				[logger.info(f"{value}") for value in score]
			else:
				logger.info(f"{key}: {score}")

		if results["fine_logit_margin_scores"] is not None:
			logger.info(f"Fine logits margin scores:")
			for key, score in results["fine_logit_margin_scores"].items():
				if key in ["Correct Margins", "Incorrect Margins"]:
					logger.info(f"{key}:")
					[logger.info(f"{value}") for value in score]
				else:
					logger.info(f"{key}: {score}")

		else: logger.info("Fine head not trained, so no fine logit margins.")

	logger.info("Classification report:\n%s", results["classification_report"])


def main() -> None:
	"""
	Run the model analysis pipeline.

	Parses the command-line arguments, selects the requested model and
	dataset, loads the corresponding checkpoint and test data, extracts
	logits and embeddings, computes predictions and analysis metrics,
	logs the results, and generates a UMAP projection.

	Returns
	-------
	None
	"""
	args = initiate_parser()
	logger = create_logger(args)

	if args.dataset in ["ILDC_single", "ILDC_multi"] or args.load_lc_ckpt == "y": num_labels = 2
	else: num_labels = 3

	model_choices = {
		"bert"        : "google-bert/bert-base-uncased",
		"legalbert"   : "nlpaueb/legal-bert-base-uncased",
		"inlegalbert" : "law-ai/InLegalBERT",
		"llama"       : "meta-llama/Llama-3.1-8B",
		"mistral"     : "mistralai/Mistral-7B-v0.3",
		"qwen"        : "Qwen/Qwen2.5-7B",
		"saul"        : "Equall/Saul-7B-Base",
	}
	model_name = model_choices[args.model]

	dataset_choices = {
		"ILDC_single": DATASET / "ILDC_single.jsonl",
		"ILDC_multi":  DATASET / "ILDC_multi.jsonl",
		"SCIJuDAC":    DATASET / "SCIJuDAC.jsonl",
	}
	dataset_path = dataset_choices[args.dataset]

	print(f"Analysing {args.model} model trained on {args.dataset}...")
	logger.info(f"Analysing {args.model} model trained on {args.dataset}...")

	loader = DataAndModelLoader(
		model_name=model_name,
		dataset_path=dataset_path,
		num_labels=num_labels,
		batch_size=args.batch_size,
		seed=args.seed,
		load_lc_ckpt=args.load_lc_ckpt,
		load_cc_ckpt=args.load_cc_ckpt,
		load_full_ckpt=args.load_ckpt,
		enable_quantization=args.enable_quant,
	)

	model, tokenizer = loader.load_model()
	test_data = loader.load_dataset(tokenizer)
	logits, coarse_logits, fine_logits, embeddings, labels = loader.extract_features(model, tokenizer, test_data)
	probs, preds, labels = loader.get_preds(logits, coarse_logits, fine_logits, labels)

	if args.load_lc_ckpt == "y": c = "lc"
	elif args.load_cc_ckpt =="y": c = "cc"
	else: c = "exp" 
	filename = f"{c}_{args.model}_{args.dataset}"

	analysis = Analysis(
		logits=logits,
		coarse_logits=coarse_logits,
		fine_logits=fine_logits,
		probs=probs,
		preds=preds,
		labels=labels,
		embeddings=embeddings,
		collapsed_classifier=args.load_cc_ckpt,
		filename=filename
	)

	results = analysis.run()
	log_metrics(logger, results, args.load_cc_ckpt)
	plot_umap(filename, embeddings, labels, args)


if __name__ == '__main__':
	main()
