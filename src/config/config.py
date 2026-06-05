"""
Contains path roots for all the folders in the project.
Individual path variables can be imported directly into the code as required.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# -> Level 1
CHECKPOINTS_ROOT = ROOT / "checkpoints"


# -> Level 1
DATA_ROOT = ROOT / "data"
# --> Level 2
DATASET = DATA_ROOT / "dataset"
CONFUSION_MATRICES = DATA_ROOT / "confusion_matrices"

# --> Level 2
PARAWISE_DATA = DATA_ROOT / "parawise_data"

# --> Level 2
SCRAPED_DATA = DATA_ROOT / "scraped_data"
# ---> Level 3
HTML_DATA = SCRAPED_DATA / "html"
TEXT_DATA = SCRAPED_DATA / "text"


# -> Level 1
IMAGES_ROOT = ROOT / "images"


# -> Level 1
LOGS_ROOT = ROOT / "logs"
# --> Level 2
ANALYSIS = LOGS_ROOT / "analysis"
# ---> Level 3
COLLAPSED_LABELS_LOGS = ANALYSIS / "collapsed_labels"
SEPARABILITY_AND_CALIBRATION_LOGS = ANALYSIS / "separability_and_calibration"

# --> Level 2
ILDC_LOGS = LOGS_ROOT / "ILDC"
SCIJUDAC_LOGS = LOGS_ROOT / "SCIJuDAC"


# -> Level 1
SOURCE = ROOT / "src"
# --> Level 2
DATASET_BUILDER = SOURCE / "dataset_builder"
LABELLING = SOURCE / "labelling"
MODELS = SOURCE / "models"
SCRAPER = SOURCE / "scraper"
TOOLS = SOURCE / "tools"
UTILS = SOURCE / "utilities"
