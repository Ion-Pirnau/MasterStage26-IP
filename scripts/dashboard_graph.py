import os, sys

sys.path.append(os.path.dirname(__file__))

from utils.other_utilities import plot_training_metrics, plot_validation_metrics, plot_loss_by_timestep_bin
from utils.utils_method import get_full_path
from utils.constants import PATH_MODEL_SDXL_ORI


if __name__ == "__main__":
    loss_history_path = os.path.join(
            get_full_path(PATH_MODEL_SDXL_ORI),
            "loss_history.csv"
        )

    loss_bin_path = os.path.join(
                get_full_path(PATH_MODEL_SDXL_ORI),
                "loss_history_bins.csv"
    )

    loss_validation_path = os.path.join(
            get_full_path(PATH_MODEL_SDXL_ORI),
            "validation_history.csv"
        )

    #plot_training_metrics(csv_path=loss_history_path)

    #plot_loss_by_timestep_bin(csv_path=loss_bin_path)

    plot_validation_metrics(csv_path=loss_validation_path)


