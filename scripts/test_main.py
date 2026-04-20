from utils.utils_method import *


if __name__ == "__main__":

    path_asc = get_full_path("data", "heightmaps", "heightmap_0000.asc")
    
    if path_exists(path_asc):
        print(f"{path_asc} ESISTE")
    else:
        print("Errore")

    v, f = mesh_from_asc(path_asc)

    path_off = get_full_path("data", "meshes", "test.off")
    print(path_off)
    save_off(v, f, path_off)
