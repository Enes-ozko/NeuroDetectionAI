import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Charge les paramètres depuis le fichier YAML."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def ensure_dirs_exist(config):
    """Vérifie que les dossiers de données existent."""
    for dir_path in config['paths'].values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)