import json
import os
import logging

def carregar_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error (f"Não encontrado o arquivo: {file}")
        return None
    

def evaluate_datadrift(file, min_datashare): 
    datadrift = False
    data = carregar_json(file) 

    for metric in data.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            #drift_share = metric.get("result", {}).get("drift_share")
            share_of_drifted_columns = metric.get("result", {}).get("share_of_drifted_columns")

    if share_of_drifted_columns > min_datashare:
        logging.info(f"Drift de dados Detectado. Share of drifted columns ({share_of_drifted_columns}) maior que o mínimo drift_share definido ({min_datashare})")
        datadrift = True
    else:
        logging.info("Nenhum drift de dados detectado.")
        logging.info(f"Modelo eficiente.  Share of drifted columns ({share_of_drifted_columns}) menor que o mínimo drift_share definido ({min_datashare})")

    return datadrift

def evaluate_classdrift(file, min_f1): 
    classdrift = False
    data = carregar_json(file) 


    f1_score_current = None

    for metric in data.get("metrics", []):
        if metric.get("metric") == "ClassificationQualityMetric":
            current_f1 = metric.get("result", {}).get("current", {}).get("f1")
            if current_f1 is not None:
                f1_score_current = current_f1
            break

    if f1_score_current is not None :

        if f1_score_current < min_f1:
            logging.info(f"Drift de classificação Detectado. F1 corrente ({f1_score_current}) menor que o mínimo ({min_f1})")
            classdrift = True
        else:
            logging.info("Nenhum drift de classificação detectado.")
            logging.info(f"Modelo eficiente. F1 corrente ({f1_score_current}) maior que o mínimo ({min_f1})")

    else:
        logging.error("F1 Score não encontrado no relatório.")

    return classdrift

# Main
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

logging.info("Iniciando verificação de trigger para retreinamento do modelo")

min_f1_score = config["drift"]["min_f1_score"]
min_data_share = config["drift"]["min_data_share"]
commandRetrainig= "python " + config["drift"]["code_retraining"]

logging.info(f"Mínimo F1-Score: {min_f1_score} e Mínimo Data-share de colunas: {min_data_share}")
             
file_json_classdrift = config["files"]["classdrift_json"]
file_json_datadrift = config["files"]["datadrift_json"]

logging.info (f'Json para datadrift: {file_json_datadrift}')
logging.info (f'Json para classdrift: {file_json_classdrift}')

avalia_classdrift = evaluate_classdrift(file_json_classdrift, min_f1_score)
avalia_datadrift = evaluate_datadrift(file_json_datadrift, min_data_share)
if (avalia_classdrift or avalia_datadrift):
    logging.info(f"Iniciando o retreinamento do modelo: {commandRetrainig}")
    os.system(commandRetrainig)