"""Contains paths of directories."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] # Root directory Path

# -> Level 1
DATA = BASE_DIR / "data"

# --> Level 2
BATCHES = DATA / "batches"
# ---> Level 3
BATCH_INPUT = BATCHES / "input"
BATCH_VALIDATION = BATCHES / "validation"

# --> Level 2
DATASET = DATA / "dataset"

# --> Level 2
INTERIM = DATA / "interim"
# ---> Level 3
PARAWISE_DATA = INTERIM / "parawise_data"
LABEL_OUTPUT = INTERIM / "label_output"

# --> Level 2
LABELLED_JUDGMENTS = DATA / "labelled_judgments"


# -> Level 1
LOGS = BASE_DIR / "logs"
# --> Level 2
LABEL_GENERATION_LOGS = LOGS / "label_generation"
EXP_LOGS = LOGS / "exp"


# -> Level 1
PROMPTS = BASE_DIR / "prompts"


# -> Level 1
SOURCE_FILES = BASE_DIR / "src"
# --> Level 2
BATCHING = SOURCE_FILES / "batching"
DATASET_BUILDER = SOURCE_FILES / "dataset_builder"
FILE_WRITERS = SOURCE_FILES / "file_writers"
LABELLING = SOURCE_FILES / "labelling"
TOOLS = SOURCE_FILES / "tools"
