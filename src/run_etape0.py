import time
import sys
import traceback

# Importation des fonctions de traitement depuis le sous-dossier 'src.data'
from src.data.step01_dicom_to_nifti import importer_et_convertir
from src.data.step02_skull_stripping import enlever_crane
from split_data import repartir_donnees

def main():

        print("ÉTAPE 1")
        importer_et_convertir()

        print("ÉTAPE 2")
        enlever_crane()
   
        print("ÉTAPE 3")
        from src.data.step03_extract_2D import traiter_et_extraire
        traiter_et_extraire()

        print("SÉPARATION ET REPARTITION DES DONNÉES (Train 80% / Test 20%)")
        repartir_donnees()

if __name__ == "__main__":
    main()