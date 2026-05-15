import time
import sys
import traceback

# Importation de tes fonctions depuis le sous-dossier 'data'
from data.step01_dicom_to_nifti import importer_et_convertir
from data.step02_skull_stripping import enlever_crane
from data.step03_extract_2D import traiter_et_extraire

def main():
    print("=======================================================")
    print("🚀 DÉMARRAGE DU PIPELINE DE PRÉPARATION DES DONNÉES 🚀")
    print("=======================================================\n")

    start_time = time.time()

    try:
        # --- ÉTAPE 1 ---
        print(">>> LANCEMENT DE L'ÉTAPE 1 : Import & Conversion")
        importer_et_convertir()
        print("-" * 55 + "\n")

        # --- ÉTAPE 2 ---
        print(">>> LANCEMENT DE L'ÉTAPE 2 : Skull Stripping")
        enlever_crane()
        print("-" * 55 + "\n")

        # --- ÉTAPE 3 ---
        print(">>> LANCEMENT DE L'ÉTAPE 3 : Extraction 2D & Traitement")
        traiter_et_extraire()
        print("-" * 55 + "\n")

        # --- FIN ---
        end_time = time.time()
        duree_minutes = (end_time - start_time) / 60

        print("=======================================================")
        print(f"🎉 PIPELINE TERMINÉ AVEC SUCCÈS en {duree_minutes:.2f} minutes !")
        print(f"➡️ Tes données sont prêtes dans : data/03_processed/")
        print("=======================================================")

    except BaseException as e:
        # En cas de crash (manque de RAM, fichier corrompu, etc.)
        print("\n" + "=" * 55)
        print("❌ ERREUR CRITIQUE PENDANT L'EXÉCUTION DU PIPELINE :")
        print("=" * 55)
        traceback.print_exc()  # Affiche exactement où le code a planté
        sys.exit(1)

if __name__ == "__main__":
    main()