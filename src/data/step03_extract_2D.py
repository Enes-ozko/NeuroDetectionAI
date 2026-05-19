import os
import nibabel as nib
import numpy as np
import cv2

DIR_INTERIM = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\02_interim"
DIR_PROCESSED = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\03_processed"

TAILLE_CIBLE = (224, 224)
MAX_PATIENTS_TEST = None  
PAS_SLICE = 2

def normaliser_0_255(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    norm = (image - min_val) / (max_val - min_val)
    return (norm * 255.0).astype(np.uint8)

def redimensionner_sans_deformer(image):
    h, w = image.shape
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim), dtype=image.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = image
    return cv2.resize(square, TAILLE_CIBLE, interpolation=cv2.INTER_AREA)

def traiter_et_extraire():
    print("Etape 3 : Extraction 2D JPG (Filtre anatomique Hypophyse et casse sécurisés)")
    os.makedirs(DIR_PROCESSED, exist_ok=True)

    fichiers_t1 = [f for f in os.listdir(DIR_INTERIM) if 't1.nii' in f.lower()]
    patients_traites = 0

    for fichier_t1 in fichiers_t1:
        if MAX_PATIENTS_TEST is not None and patients_traites >= MAX_PATIENTS_TEST:
            break

        nom_inferieur = fichier_t1.lower()
        patient_id = fichier_t1
        for suffixe in ['_t1.nii.gz', '_t1.nii', '-t1.nii.gz', '-t1.nii']:
            if suffixe in nom_inferieur:
                idx = nom_inferieur.index(suffixe)
                patient_id = fichier_t1[:idx]
                break
        else:
            patient_id = fichier_t1.replace('.nii.gz', '').replace('.nii', '')

        chemin_t1 = os.path.join(DIR_INTERIM, fichier_t1)

        chemin_mask = os.path.join(DIR_INTERIM, f"{patient_id}_mask.nii.gz")
        if not os.path.exists(chemin_mask):
            chemin_mask = os.path.join(DIR_INTERIM, f"{patient_id}_mask.nii")

        a_un_masque = os.path.exists(chemin_mask)
        print(f"Patient : {patient_id} | Masque détecté : {a_un_masque}")

        img_t1 = nib.load(chemin_t1)
        img_t1_canonical = nib.as_closest_canonical(img_t1)
        vol_t1 = img_t1_canonical.get_fdata()

        if a_un_masque:
            img_mask = nib.load(chemin_mask)
            if img_mask.shape == img_t1.shape:
                img_mask = nib.Nifti1Image(img_mask.dataobj, img_t1.affine, img_t1.header)
            img_mask_canonical = nib.as_closest_canonical(img_mask)
            vol_mask = img_mask_canonical.get_fdata()

        nb_slices = vol_t1.shape[2]
        slices_valides = 0
        max_vol = np.max(vol_t1)

        for z in range(0, nb_slices, PAS_SLICE):
            slice_t1 = vol_t1[:, :, z].astype(np.float32)
            conserver_slice = False
     
            if patient_id.startswith("Gliome_") or patient_id.startswith("Meningiome_"):
                if a_un_masque:
                    slice_mask = vol_mask[:, :, z].astype(np.uint8)
                    if np.sum(slice_mask) > 0:
                        conserver_slice = True
                else:
                    pixels_cerveau = np.sum(slice_t1 > (max_vol * 0.15))
                    if (pixels_cerveau / (slice_t1.shape[0] * slice_t1.shape[1])) >= 0.15:
                        conserver_slice = True
            
            elif patient_id.startswith("Hypophyse_"):
                limite_basse = int(nb_slices * 0.22)
                limite_haute = int(nb_slices * 0.45)
                if limite_basse <= z <= limite_haute:
                    pixels_cerveau = np.sum(slice_t1 > (max_vol * 0.15))
                    if (pixels_cerveau / (slice_t1.shape[0] * slice_t1.shape[1])) >= 0.10:
                        conserver_slice = True
            
            elif patient_id.startswith("Sain_"):
                pixels_cerveau = np.sum(slice_t1 > (max_vol * 0.15))
                total_pixels = slice_t1.shape[0] * slice_t1.shape[1]
                if (pixels_cerveau / total_pixels) >= 0.15:
                    conserver_slice = True

            if conserver_slice:
                pixels_visibles = np.sum(slice_t1 > (max_vol * 0.15))
                if np.max(slice_t1) == 0 or pixels_visibles < 500:
                    conserver_slice = False

            if conserver_slice:
                slice_t1_resized = redimensionner_sans_deformer(slice_t1)
                slice_t1_final = normaliser_0_255(slice_t1_resized)

                nom_t1 = f"{patient_id}_z{z:03d}_t1.jpg"
                cv2.imwrite(os.path.join(DIR_PROCESSED, nom_t1), slice_t1_final)
                slices_valides += 1

        print(f"Patient {patient_id} : {slices_valides} coupes extraites.")
        patients_traites += 1

if __name__ == "__main__":
    traiter_et_extraire()