import os
import shutil
import subprocess

DIR_RAW_SOURCE = r"C:\Users\munar\Desktop\dataset"
DIR_RAW_PROJET = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\01_raw"
DCM2NIIX_PATH = r"C:\chemin\vers\dcm2niix.exe"

def importer_et_convertir():
    print("Etape 1 : Importation récursive avec filtrage T1, Masques et classe Sain")
    os.makedirs(DIR_RAW_PROJET, exist_ok=True)

    for racine, dossiers, fichiers in os.walk(DIR_RAW_SOURCE):
        for fichier in fichiers:
            chemin_source = os.path.join(racine, fichier)
            fichier_lower = fichier.lower()
            racine_lower = racine.lower()
            if "t2" in fichier_lower or "flair" in fichier_lower or "dwi" in fichier_lower:
                continue
            if "pituitary" in racine_lower or "hypophyse" in racine_lower:
                if "prediction" in fichier_lower or "segmentationseed" in fichier_lower:
                    continue
                if "cecor" in fichier_lower or "cesag" in fichier_lower or "_cor_" in fichier_lower:
                    continue
                if "ce3dnavigation" not in fichier_lower and "groundtruth" not in fichier_lower:
                    continue

            nom_propre = fichier
            nom_propre = nom_propre.replace("-seg.nii", "_mask.nii")
            nom_propre = nom_propre.replace("_gtv.nii", "_mask.nii")
            nom_propre = nom_propre.replace("_label-groundTruth.nii", "_mask.nii")
            nom_propre = nom_propre.replace("-t1n.nii", "_t1.nii")
            nom_propre = nom_propre.replace("-t1c.nii", "_t1.nii")
            nom_propre = nom_propre.replace("_t1c.nii", "_t1.nii")
            nom_propre = nom_propre.replace("_acq-CE3DNavigation_T1w.nii", "_t1.nii")
            nom_propre = nom_propre.replace("-.gz_t1.nii", "_t1.nii.gz")
            nom_propre = nom_propre.replace("_.gz_t1.nii", "_t1.nii.gz")
            nom_propre = nom_propre.replace(".gz_t1.nii", "_t1.nii.gz")
            nom_propre = nom_propre.replace("-.gz_mask.nii", "_mask.nii.gz")
            nom_propre = nom_propre.replace("_.gz_mask.nii", "_mask.nii.gz")
            nom_propre = nom_propre.replace(".gz_mask.nii", "_mask.nii.gz")

            if "brats-gli" in nom_propre.lower() or "gliome" in racine_lower:
                if not nom_propre.startswith("Gliome_"):
                    nom_propre = "Gliome_" + nom_propre
            elif "brats-men" in nom_propre.lower() or "meningiome" in racine_lower:
                if not nom_propre.startswith("Meningiome_"):
                    nom_propre = "Meningiome_" + nom_propre
            elif "pituitary" in racine_lower or "hypophyse" in racine_lower:
                if not nom_propre.startswith("Hypophyse_"):
                    nom_propre = "Hypophyse_" + nom_propre
            elif "sain" in racine_lower or "sain" in fichier_lower or "healthy" in racine_lower or "control" in racine_lower or "ixi" in fichier_lower:
                if not nom_propre.startswith("Sain_"):
                    nom_propre = "Sain_" + nom_propre

            chemin_dest = os.path.join(DIR_RAW_PROJET, nom_propre)

            if fichier.endswith('.nii') or fichier.endswith('.nii.gz'):
                if not os.path.exists(chemin_dest):
                    shutil.copy2(chemin_source, chemin_dest)
                    print(f"Importé et classé : {nom_propre}")

if __name__ == "__main__":
    importer_et_convertir()