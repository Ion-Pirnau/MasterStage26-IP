import os, sys

sys.path.append(os.path.dirname(__file__))

from utils.other_utilities import plot_training_metrics
from utils.utils_method import get_full_path
from utils.constants import PATH_MODEL_SDXL_ORI


if __name__ == "__main__":
    loss_history_path = os.path.join(
            get_full_path(PATH_MODEL_SDXL_ORI),
            "loss_history.csv"
        )

    plot_training_metrics(csv_path=loss_history_path)


