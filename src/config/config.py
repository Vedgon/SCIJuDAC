from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --> Level 1
CHECKPOINTS_ROOT = ROOT / "checkpoints"


# --> Level 1
DATA_ROOT = ROOT / "data"

# --> Level 2
DATASET = DATA_ROOT / "dataset"

# --> Level 2
PARAWISE_DATA = DATA_ROOT / "parawise_data"

# --> Level 2
SCRAPED_DATA = DATA_ROOT / "scraped_data"
# -> Level 3
HTML_DATA = SCRAPED_DATA / "html"
TEXT_DATA = SCRAPED_DATA / "text"


# --> Level 1
LOGS_ROOT = ROOT / "logs"


# --> Level 1
SOURCE = ROOT / "src"
# --> Level 2
DATASET_BUILDER = SOURCE / "dataset_builder"
LABELLING = SOURCE / "labelling"
MODELS = SOURCE / "models"
PREPROCESSING = SOURCE / "preprocessing"
SCRAPER = SOURCE / "scraper"
TOOLS = SOURCE / "tools"
UTILS = SOURCE / "utilities"
