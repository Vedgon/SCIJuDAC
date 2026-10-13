import sys
from pathlib import Path
src_path = Path(__file__).resolve().parents[2]
sys.path.append(str(src_path))

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
	confusion_matrix,
	silhouette_score,
	classification_report,
	davies_bouldin_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import IMAGES_ROOT, CONFUSION_MATRICES
from torch import Tensor
from numpy import ndarray


class Analysis():
	"""
	Analysis pipeline for computing different separability and calibration
	scores. The following metrics are computed:
	- Confusion matrix
	- Brier score
	- Expected Calibration Error
	- Silhouette Score
	- Top-2 logit margin
	"""
	def __init__(
				self, logits:Tensor, coarse_logits:Tensor, fine_logits:Tensor,
				probs:Tensor, preds:Tensor|list[Tensor],
				labels:Tensor, embeddings:Tensor,
				collapsed_classifier:str, filename:Path
	) -> None:
		"""
		Initializes the analysis object.

		Arguments
		---------
		num_labels : int
			Total number of labels in the dataset.

		coarse_logits : Tensor
			Logits produced by the coarse-grained classification head.
			Shape: (num_samples, num_coarse_labels)

		fine_logits : Tensor | None
			Logits produced by the fine-grained classification head.
			Is None if `num_labels=2`.
			Shape: (num_samples, num_fine_labels)

		probs : Tensor
			Class probabilities derived from the model logits.
			Shape: (num_samples, num_labels)

		preds : Tensor | list[Tensor]
			Predicted class labels.
			Shape: (num_samples,)

		labels : Tensor
			Ground-truth labels corresponding to the predictions.
			Shape: (num_samples,)

		embeddings : Tensor
			Embedding representations extracted from the model.
			Shape: (num_samples, embedding_dim)

		filename : Path
			Base filename used when saving plots and confusion matrices.

		Returns
		-------
		None
		"""
		self.embeddings = embeddings
		self.logits = logits
		self.coarse_logits = coarse_logits
		self.fine_logits = fine_logits
		self.probs = probs
		self.preds = preds
		self.labels = labels
		self.load_cc_ckpt = collapsed_classifier
		self.filename = filename


	def plot_confusion_matrix(self, cm:ndarray) -> None:
		"""
		Helper function to plot the confusion matrix.

		Arguments
		---------
		cm : ndarray
			Numpy array containing the confusion matrix.

		Returns
		-------
		None
		"""
		plt.figure(figsize=(6,5))
		sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis")
		plt.xlabel("Predicted")
		plt.ylabel("True")
		plt.title("Confusion Matrix")
		output_path = IMAGES_ROOT / f"cm_{self.filename}.png"
		output_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path)
		plt.close()


	def generate_confusion_matrix(self, y_pred:Tensor, y_true:Tensor) -> None:
		"""
		Computes confusion matrix for the given predictions and labels.

		Arguments
		---------
		y_pred : Tensor
			The predicted classes.
			Shape : (num_samples, num_labels)

		y_true : Tensor
			Actual classes.
			Shape : (num_samples,)

		Returns
		-------
		None
		"""
		classes = [i for i in range(0, self.num_labels)]
		cm = confusion_matrix(y_true, y_pred, labels=classes) # , normalize="true"
		cm_filepath = CONFUSION_MATRICES / f"cm_{self.filename}.npy"
		cm_filepath.parent.mkdir(parents=True, exist_ok=True)
		np.save(cm_filepath, cm)
		self.plot_confusion_matrix(cm)


	def compute_brier_score(self, probs:Tensor, labels:Tensor) -> float:
		"""
		Computes the multiclass Brier score.
		The Brier score measures the mean squared difference between the
		predicted class probabilities and the one-hot encoded ground-truth
		labels. Lower values indicate better calibration.

		Arguments
		---------
		probs : Tensor
			Predicted class probabilities.
			Shape: (num_samples, num_classes)

		labels : Tensor
			Ground-truth class labels.
			Shape: (num_samples,)

		Returns
		-------
		brier_score : float
			Multiclass Brier score.
		"""
		num_classes = probs.shape[1]
		# print("BS # of classes:", num_classes, "BS classes:", list(set(labels)))
		one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
		one_hot[np.arange(len(labels)), labels] = 1
		brier_score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

		return brier_score.item()


	@staticmethod
	def compute_ece(probs:Tensor, labels:Tensor, n_bins:int=15) -> float:
		"""
		Computes the Expected Calibration Error (ECE).
		ECE measures the discrepancy between model confidence and empirical
		accuracy by partitioning predictions into confidence bins.

		Arguments
		---------
		probs : Tensor
			Predicted class probabilities.
			Shape: (num_samples, num_classes)

		labels : Tensor
			Ground-truth class labels.
			Shape: (num_samples,)

		n_bins : int, default=15
			Number of confidence bins used during calibration analysis.

		Returns
		-------
		ece : float
			Value of Expected Calibration Error.
		"""
		confidences, predictions = torch.max(probs, dim=1)
		correct = (predictions == labels).float()

		bins = torch.linspace(0, 1, n_bins + 1)
		ece = torch.tensor(0.0)

		for i in tqdm(range(n_bins)):
			mask = (confidences > bins[i]) & (confidences <= bins[i+1])
			bin_size = mask.sum()

			if mask.any():
				acc = correct[mask].mean()
				conf = confidences[mask].mean()
				ece += torch.abs(acc - conf) * (bin_size.float() / len(probs))

		return ece


	def compute_silhouette_score(self, embeddings:Tensor) -> float:
		"""
		Computes the silhouette score of the embedding space.
		The score is computed using pairwise cosine distances between
		embeddings. Higher values indicate better cluster separation.

		Arguments
		---------
		embeddings : Tensor
			Model embeddings used for clustering analysis.
			Shape: (num_samples, embedding_dim)

		Returns
		-------
		score : float
			Silhouette score.
		"""
		D = cosine_distances(embeddings)
		D = np.clip(D, 0.0, 2.0) # Done to exclude outlier values
		score = silhouette_score(D, self.labels, metric='precomputed')

		return score


	@staticmethod
	def compute_margins(logits, labels):
		"""
		Computes top-2 logit margin statistics.
		The margin is defined as the difference between the largest and
		second-largest logit for each prediction.

		Arguments
		---------
		logits : Tensor | None
			Model logits (coarse/fine). None if `num_labels=2`
			Shape: (num_samples, num_classes)

		labels : Tensor
			Ground-truth class labels.
			Shape: (num_samples,)

		Returns
		-------
		dict[str, float | Tensor] | None
			Dictionary containing aggregate margin statistics and the
			individual margins for correctly and incorrectly classified
			examples.

			Returns ``None`` if no logits are provided.
		"""
		if logits is None or logits.numel() == 0: return None
		sorted_logits, _ = torch.sort(logits, dim=1)
		margins = sorted_logits[:, -1] - sorted_logits[:, -2]

		predictions = torch.argmax(logits, dim=1)
		correct = (predictions == labels)

		return {
			"Mean Margin": margins.mean().item(),
			"Mean Correct Margin": margins[correct].mean().item() if correct.any() else 0.0,
			"Mean Incorrect Margin": margins[~correct].mean().item() if (~correct).any() else 0.0,
			"Correct Margins": margins[correct],
			"Incorrect Margins": margins[~correct]
		}


	def evaluate(self)-> dict:
		"""
		Computes all analysis metrics.

		Returns
		-------
		dict
			Dictionary containing calibration, separability, and
			classification metrics, including:
			- Brier score
			- Expected Calibration Error
			- Silhouette score
			- Coarse logit margins
			- Fine logit margins
			- Classification report
		"""
		if self.load_cc_ckpt == "y":
			return {
				"brier_score": self.compute_brier_score(self.probs, self.labels),
				"ece_score": self.compute_ece(self.probs, self.labels),
				"silhouette_value": self.compute_silhouette_score(self.embeddings),
				"davies_bouldin_index": davies_bouldin_score(self.embeddings, self.labels),
				"logit_margin_scores": self.compute_margins(self.logits, self.labels),
				"classification_report": classification_report(self.labels, self.preds, digits=16)
			}

		else:
			return {
				"brier_score": self.compute_brier_score(self.probs, self.labels),
				"ece_score": self.compute_ece(self.probs, self.labels),
				"silhouette_value": self.compute_silhouette_score(self.embeddings),
				"davies_bouldin_index": davies_bouldin_score(self.embeddings, self.labels),
				"coarse_logit_margin_scores": self.compute_margins(self.coarse_logits, self.labels),
				"fine_logit_margin_scores": self.compute_margins(self.fine_logits, self.labels),
				"classification_report": classification_report(self.labels, self.preds, digits=16)
			}


	def run(self):
		"""
		Executes the complete analysis pipeline.
		Generates and saves the confusion matrix, then computes all
		evaluation metrics.

		Returns
		-------
		dict
			Dictionary containing all computed analysis metrics.
		"""
		self.generate_confusion_matrix(self.preds, self.labels)
		metrics = self.evaluate()

		return metrics
