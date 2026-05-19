import os
import random
import shutil
from collections import defaultdict
from src.data.step03_extract_2D import traiter_et_extraire

DIR_PROCESSED = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\03_processed"

def repartir_donnees():
    fichiers = [f for f in os.listdir(DIR_PROCESSED) if f.endswith('_t1.jpg')]
    
    patients_par_classe = defaultdict(set)
    fichiers_par_patient = defaultdict(list)
    
    for f in fichiers:
        classe = f.split('_')[0]
        patient_id = f.split('_z')[0]
        patients_par_classe[classe].add(patient_id)
        fichiers_par_patient[patient_id].append(f)
        
    for classe, patients in patients_par_classe.items():
        liste_patients = list(patients)
        random.seed(42)
        random.shuffle(liste_patients)
        
        split_idx = int(len(liste_patients) * 0.8)
        patients_train = liste_patients[:split_idx]
        patients_test = liste_patients[split_idx:]
        
        dossier_train = os.path.join(DIR_PROCESSED, "train", classe)
        dossier_test = os.path.join(DIR_PROCESSED, "test", classe)
        
        os.makedirs(dossier_train, exist_ok=True)
        os.makedirs(dossier_test, exist_ok=True)
        
        for p in patients_train:
            for f in fichiers_par_patient[p]:
                shutil.move(os.path.join(DIR_PROCESSED, f), os.path.join(dossier_train, f))
                
        for p in patients_test:
            for f in fichiers_par_patient[p]:
                shutil.move(os.path.join(DIR_PROCESSED, f), os.path.join(dossier_test, f))

if __name__ == "__main__":
    traiter_et_extraire()
    print("Repartition des donnees (Train 80% / Test 20%)")
    repartir_donnees()
