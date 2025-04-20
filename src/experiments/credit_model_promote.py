import json
import logging

import mlflow
from mlflow.tracking import MlflowClient

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


client = MlflowClient(tracking_uri=config["mlflow"]["mlflow_uri"])

# Lista dos modelos a serem considerados
model_names = [
    "CreditCard_RandomForest",
    "CreditCard_XGBoost",
    "CreditCard_LogisticRegression"
]

staging_threshold = config["experiments"]["staging_threshold"]  # F1-score mínimo para ir para Staging
logging.info(f"Iniciando promoção do modelo campeão. F1-score mínimo para ir para Staging (staging Threshold): {staging_threshold}")

best_model_info = {
    "model_name": None,
    "version": None,
    "f1_score": 0
}

for model_name in model_names:
    versions = client.search_model_versions(f"name='{model_name}'")

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics

        if "f1_score" in metrics:
            f1 = metrics["f1_score"]

            # Se atender ao threshold, mover para Staging
            if f1 > staging_threshold:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Staging"
                )
                logging.info(f"Modelo {model_name} versão {version.version} com F1-score {f1} movido para Staging.")

            # Verifica se este é o melhor modelo até agora
            if f1 > best_model_info["f1_score"]:
                best_model_info = {
                    "model_name": model_name,
                    "version": version.version,
                    "f1_score": f1
                }

# Promover o melhor modelo entre todos para Production
if best_model_info["model_name"]:
    client.transition_model_version_stage(
        name=best_model_info["model_name"],
        version=best_model_info["version"],
        stage="Production"
    )
    logging.info(f"Modelo {best_model_info['model_name']} versão {best_model_info['version']} agora é o campeão com F1-score de {best_model_info['f1_score']}.")
else:
    logging.info("Nenhum modelo atende ao critério para ser campeão.")
