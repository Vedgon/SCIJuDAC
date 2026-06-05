"""
Fetches the path of the latest checkpoint of the model/the
root directory where the model checkpoints will be stored.
"""
from typing import TYPE_CHECKING, Literal
from pathlib import Path
if TYPE_CHECKING:
	from pathlib import Path

def get_checkpoint_path(
		checkpoint_root:Path, dataset_path:Path, model_name:str,
		collapse_labels:Literal["y", "n"]="n",
		choice:Literal["y", "n"]="n"
	) -> Path:
	"""
	Returns the path of the latest checkpoint of the model if
	the `choice` flag is set to "y", else returns the root
	directory path of the folder where the checkpoint
	is to be stored.

	Arguments
	---------
	checkpoint_root : Path
		Root path of the checkpoint folder.

	dataset_path : Path
		Path to the dataset file.

	model_name : str
		Name of the model that's been loaded.

	collapse_labels : Literal["y", "n"]
		Flag to determine whether ablation by collapsing
		labels in SCIJuDAC is enabled or not.

	choice : Literal["y", "n"]
		Flag to determine whether the checkpoint is to be
		loaded or not.

	Returns
	-------
	checkpoint_path : Path
		Path of the folder where the checkpoint is stored or
		is loaded from, depending on the user `choice`.
	"""
	dataset_name = dataset_path.stem
	model = model_name.split("/")[-1]
	if collapse_labels=="y": checkpoint_output_path = checkpoint_root / dataset_name / "ablation" / model
	else: checkpoint_output_path = checkpoint_root / dataset_name / "exp" / model
	checkpoint_output_path.mkdir(parents=True, exist_ok=True)

	if choice == "y":
		checkpoints = [checkpoint_path for checkpoint_path in checkpoint_output_path.iterdir() if checkpoint_path.is_dir()]
		if checkpoints == []: return checkpoint_output_path
		latest_checkpoint_val = 0
		checkpoint_idx = 0

		for idx, checkpoint_path in enumerate(checkpoints):
			current_checkpoin_val = int(checkpoint_path.stem.split("-")[-1])
			if (current_checkpoin_val >= latest_checkpoint_val):
				latest_checkpoint_val = current_checkpoin_val
				checkpoint_idx = idx

		checkpoint_path = checkpoints[checkpoint_idx]

		return checkpoint_path

	else:
		return checkpoint_output_path
