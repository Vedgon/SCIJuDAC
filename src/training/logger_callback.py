import logging
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
	def __init__(self, logger):
		self.logger = logger

	def on_log(self, args, state, control, logs=None, **kwargs):
		if logs is None:
			return

		if "loss" in logs:
			self.logger.info(f"Train loss: {logs['loss']}")

		if "eval_loss" in logs:
			self.logger.info(f"Eval loss: {logs['eval_loss']}")
			self.logger.info(f"Eval MF1: {logs['eval_f1_macro']}")
			self.logger.info(f"Eval WF1: {logs['eval_f1_weighted']}")
			self.logger.info(f"Eval accuracy: {logs['eval_accuracy']}")
			self.logger.info(f"Eval runtime: {logs['eval_runtime']}\n")
