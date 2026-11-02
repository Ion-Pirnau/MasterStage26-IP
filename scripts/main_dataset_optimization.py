
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils_method import get_full_path
from utils.other_utilities import feature_analysis_heightmaps
from utils.constants import PATH_DATASET_GENERATED, PATH_DATASET_GENERATED_OPT

if __name__ == "__main__":
    feature_analysis_heightmaps(get_full_path(PATH_DATASET_GENERATED), get_full_path(PATH_DATASET_GENERATED_OPT),
                                target_images=400, n_clusters=5)


