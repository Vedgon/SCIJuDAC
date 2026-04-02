def load_checkpoint_from_disk(checkpoint_path, trainer, logger):
	if any(checkpoint_path.iterdir()):
		choice = input("Do you want to resume training from the checkpoint (Y/N): ").upper()
		while choice not in ["Y", "N"]:
			print("Incorrect choice. Please enter a valid character (Y/N).")
			choice = input("Do you want to resume training from the checkpoint (Y/N): ").upper()

		if choice == "Y":
			checkpoint = [checkpoint_path for checkpoint_path in checkpoint_path.iterdir()][-1]
			trainer.train(resume_from_checkpoint=checkpoint)
			logger.info("Resuming from checkpoint...")
		else:
			trainer.train()

	else: trainer.train()

	return trainer