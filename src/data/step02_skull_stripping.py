import os
import shutil
import ants
import antspynet

DIR_RAW = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\01_raw"
DIR_INTERIM = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\02_interim"

def enlever_crane():
    print("Etape 2 : Skull Stripping avec ANTsPyNet")
    os.makedirs(DIR_INTERIM, exist_ok=True)

    for fichier in os.listdir(DIR_RAW):
        chemin_source = os.path.join(DIR_RAW, fichier)
        chemin_dest = os.path.join(DIR_INTERIM, fichier)

        if not fichier.endswith('.nii') and not fichier.endswith('.nii.gz'):
            continue

        if "mask" in fichier:
            shutil.copy2(chemin_source, chemin_dest)
            continue

        if fichier.startswith("Sain_") or fichier.startswith("Hypophyse_"):
            if not os.path.exists(chemin_dest):
                print(f"Stripping en cours pour : {fichier}")
                img = ants.image_read(chemin_source)
                prob_brain_mask = antspynet.brain_extraction(img, modality="t1")
                mask = ants.threshold_image(prob_brain_mask, low_thresh=0.5, high_thresh=1.0)
                brain = mask * img
                ants.image_write(brain, chemin_dest)
        else:
            if not os.path.exists(chemin_dest):
                shutil.copy2(chemin_source, chemin_dest)
                print(f"Copie directe : {fichier}")

if __name__ == "__main__":
    enlever_crane()