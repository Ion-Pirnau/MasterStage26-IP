
from utils.other_utilities import converti_exr_a_png_16bit
from utils.constants import HEIGHTMAPS_EXR_LOCATION, HEIGHTMAPS_PNG_LOCATION

if __name__ == "__main__":

    converti_exr_a_png_16bit(cartella_sorgente=HEIGHTMAPS_EXR_LOCATION, cartella_destinazione=HEIGHTMAPS_PNG_LOCATION)