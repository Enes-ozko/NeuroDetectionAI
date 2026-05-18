import time
import sys
import traceback

# Importation corrigée avec le préfixe 'src.'
from src.data.step01_dicom_to_nifti import importer_et_convertir
from src.data.step02_skull_stripping import enlever_crane
from split_data import repartir_donnees

def main():
        print("ETAPE 1")
        importer_et_convertir()

        print("ETAPE 2")
        enlever_crane()

        print("ETAPE 3")
        from src.data.step03_extract_2D import traiter_et_extraire
        traiter_et_extraire()

        print("\nSPLIT")
        repartir_donnees()

if __name__ == "__main__":
    main()