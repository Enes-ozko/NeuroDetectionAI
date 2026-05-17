import os
from run_inference import run_inference

DIR_PROCESSED = r"C:\Users\munar\Desktop\NeuroDetectionAI\data\03_processed"
DIR_OUTPUT = r"outputs/batch_results"

def lancer_batch():
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    fichiers = [f for f in os.listdir(DIR_PROCESSED) if f.endswith('_t1.npy')]
    fichiers.sort()

    total = len(fichiers)
    if total == 0:
        print("Aucun fichier _t1.npy trouvé !")
        return

    print(f"Démarrage du traitement par lot : {total} coupes trouvées.\n")

    alertes_tumeur = []

    for index, fichier in enumerate(fichiers, 1):
        chemin_img = os.path.join(DIR_PROCESSED, fichier)
        nom_sortie = fichier.replace('_t1.npy', '_result.png')
        chemin_sortie = os.path.join(DIR_OUTPUT, nom_sortie)

        print(f"--- [{index}/{total}] Traitement de {fichier} ---")
        
        try:
            score = run_inference(image_path=chemin_img, save_path=chemin_sortie)
            if score is not None and score > 0.7:
                alertes_tumeur.append((fichier, score))
        except Exception as e:
            print(f"Erreur sur {fichier} : {e}")

    print("\n" + "="*60)
    print(f"TRAITEMENT TERMINÉ ! Dossier : {DIR_OUTPUT}")
    print("="*60)
    
    if len(alertes_tumeur) > 0:
        print(f"\nALERTE : {len(alertes_tumeur)} coupes présentent une probabilité de tumeur > 70% :\n")
        alertes_tumeur.sort(key=lambda x: x[1], reverse=True)
        for fichier, score in alertes_tumeur:
            print(f"  {score * 100:>5.1f}% | {fichier}")
    else:
        print("\nBilan : Aucune coupe n'a dépassé le seuil d'alerte de 70%.")

if __name__ == "__main__":
    lancer_batch()