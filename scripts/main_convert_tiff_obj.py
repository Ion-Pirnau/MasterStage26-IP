from utils.utils_method import geotiff_to_off, geotiff_to_obj, get_full_path
from utils.constants import TIFF_FOLDER, TIFF_EXTENSION, OUTPUT_TIFF_OFF_CONVERT_NAME, OBJ_EXTENSION, MESH_FOLDER

def define_complete_name(input_name, output_name):
    """
    Generates complete file names for the input GeoTIFF file and the output OBJ file
    by appending the appropriate extensions and output prefix.

    :param input_name: Base name of the input file without extension.
    :param output_name: Base name of the output file without extension.

    :return: A tuple containing:
             - complete_input_name: Full input file name with TIFF extension.
             - complete_output_name: Full output file name with OBJ extension
               and predefined output prefix.
    """

    complete_input_name = input_name+"."+TIFF_EXTENSION
    complete_output_name = OUTPUT_TIFF_OFF_CONVERT_NAME+output_name+"."+OBJ_EXTENSION

    return complete_input_name, complete_output_name


def define_path_convert(input_name, output_name):
    """
    Generates the full filesystem paths for the input GeoTIFF file and
    the output mesh file used during the conversion process.

    :param input_name: Complete name of the input GeoTIFF file.
    :param output_name: Complete name of the output mesh file.

    :return: A tuple containing:
             - cin_path: Full path to the input GeoTIFF file.
             - cout_path: Full path to the output mesh file.
    """

    cin_path = get_full_path("data", TIFF_FOLDER, input_name)

    cout_path = get_full_path("data", MESH_FOLDER, output_name)

    return cin_path, cout_path




if __name__ == "__main__":

    input_name = "sunset-crater_sanfrancisco-volcanic-field_CROP-UTM"
    output_name = "sunset-crater_sf-vf"
    

    cin_name, cout_name = define_complete_name(input_name, output_name)

    cin_path, cout_path = define_path_convert(cin_name, cout_name)

    print(cin_path)
    print(cout_path)
    
    geotiff_to_obj(cin_path, cout_path)







