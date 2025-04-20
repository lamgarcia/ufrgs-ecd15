import json
import logging

import mlflow
from mlflow.tracking import MlflowClient
import subprocess


with open("config_pipeline.json", 'r') as f:
    config = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s -  %(levelname)s - %(filename)s - %(message)s',
    handlers=[
        logging.FileHandler(config["files"]["runs_log"], mode='a'),
        logging.StreamHandler()
    ]
)

# Conexão com o MLflow
client = MlflowClient(tracking_uri=config["mlflow"]["mlflow_uri"])

# Nome do experimento
experiment_name = config["mlflow"]["mlflow_experiment"]
experiment = client.get_experiment_by_name(experiment_name)

if not experiment:
    logging.error(f"Experimento '{experiment_name}' não encontrado.")
    exit()

# ID do experimento
experiment_id = experiment.experiment_id

# Busca todos os modelos registrados
registered_models = client.search_registered_models()

# Procura o primeiro modelo em produção que tenha origem neste experimento
for model in registered_models:
    for version in model.latest_versions:
        if version.current_stage == "Production":
            run = client.get_run(version.run_id)
            if run.info.experiment_id == experiment_id:
                model_name = model.name
                logging.info(f"Modelo '{model_name}' em produção encontrado no experimento '{experiment_name}' (versão {version.version})")
                
                logging.info("Modelo servido em http://127.0.0.1:8000/invocations")
                
                # Monta URI e serve o modelo
                model_uri = f"models:/{model_name}/Production"
                subprocess.run([
                    "mlflow", "models", "serve",
                    "-m", model_uri,
                    "-p", "8000",
                    "--no-conda"
                ])
                
                exit()

logging.error(f"Nenhum modelo em produção encontrado no experimento '{experiment_name}'.")
logging.error("Não foi possível subir o serviço") 