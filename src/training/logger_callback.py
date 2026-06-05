"""
Overrides the default trainer logger to log Bicephalous-specific data.
Expected log keys include:
- loss
- coarse_loss
- fine_loss
- eval_f1_macro
- eval_f1_weighted
- eval_accuracy
- eval_runtime
"""

from transformers import(
	TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

class LoggingCallback(TrainerCallback):
	"""
	Default trainer callback override for logging training
	and evaluation metrics.

	Logs coarse- and fine-grained classification losses during training,
	as well as evaluation metrics such as macro F1, weighted F1, and
	accuracy during validation.
	"""
	def __init__(self, logger):
		"""
		Initializes the logging callback.

		Parameters
		----------
		logger : logging.Logger
			Logger instance used to record training and evaluation metrics.
		"""
		self.logger = logger


	def on_log(self,
		args: TrainingArguments,
		state: TrainerState,
		control: TrainerControl,
		logs: dict[str, float] | None = None,
		**kwargs,
	) -> None:
		"""
		Handles logging events emitted by the trainer.

		Extracts relevant training and evaluation metrics from the trainer's
		log dictionary and forwards them to the configured logger.

		Arguments
		---------
		args : TrainingArguments
			Training configuration used by the trainer.

		state : TrainerState
			Current trainer state.

		control : TrainerControl
			Trainer control object used to modify training behavior.

		logs : dict[str, float] | None, optional
			Dictionary containing metrics emitted by the trainer.

		**kwargs
			Additional keyword arguments provided by the trainer.

		Returns
		-------
		None
		"""
		if logs is None:
			return

		if "loss" in logs:
			self.logger.info(f"Total loss: {logs['loss']}")
			self.logger.info(f"Coarse loss: {logs['coarse_loss']}")
			self.logger.info(f"Fine loss: {logs['fine_loss']}")

		if "eval_f1_macro" in logs:
			self.logger.info(f"Coarse loss: {logs['coarse_loss']}")
			self.logger.info(f"Fine loss: {logs['fine_loss']}")
			self.logger.info(f"Eval MF1: {logs['eval_f1_macro']}")
			self.logger.info(f"Eval WF1: {logs['eval_f1_weighted']}")
			self.logger.info(f"Eval accuracy: {logs['eval_accuracy']}")
			self.logger.info(f"Eval runtime: {logs['eval_runtime']}\n")

		# self.logger.info(f"Step {state.global_step} | Logs: {logs}")
