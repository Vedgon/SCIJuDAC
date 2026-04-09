from pathlib import Path
def load_checkpoint_from_disk(checkpoint_path, trainer, logger):
	if any(checkpoint_path.iterdir()):
		choice = input("Do you want to resume training from the checkpoint (Y/N): ").upper()
		while choice not in ["Y", "N"]:
			print("Incorrect choice. Please enter a valid character (Y/N).")
			choice = input("Do you want to resume training from the checkpoint (Y/N): ").upper()

		if choice == "Y":
			checkpoints = [checkpoint_path for checkpoint_path in checkpoint_path.iterdir()]
			latest_checkpoint_val = 0
			checkpoint_idx = 0
			for idx, checkpoint_path in enumerate(checkpoints):
				current_checkpoin_val = int(checkpoint_path.stem.split("-")[-1])
				if (current_checkpoin_val >= latest_checkpoint_val):
					latest_checkpoint_val = current_checkpoin_val
					checkpoint_idx = idx

			checkpoint_path = checkpoints[checkpoint_idx]
			logger.info("Resuming from checkpoint...")
			trainer.train(resume_from_checkpoint=checkpoint_path)

		else:
			trainer.train()

	else: trainer.train()

	return trainer
