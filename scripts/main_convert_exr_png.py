
from utils.other_utilities import converti_exr_a_png_16bit, inspect_png
from utils.utils_method import get_full_path
from utils.constants import HEIGHTMAPS_EXR_LOCATION, HEIGHTMAPS_PNG_LOCATION, PATH_DATASET_GENERATED

if __name__ == "__main__":

    #converti_exr_a_png_16bit(cartella_sorgente=HEIGHTMAPS_EXR_LOCATION, cartella_destinazione=HEIGHTMAPS_PNG_LOCATION)
    inspect_png(get_full_path(PATH_DATASET_GENERATED, "heightmap_0000.png")
)