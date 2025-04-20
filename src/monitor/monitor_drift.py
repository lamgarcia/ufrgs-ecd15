import pandas as pd
import numpy as np
import json
import logging

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import random

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def preprocess_data(df):
    df.drop(columns=["rownames"], inplace=True, errors="ignore")
    
    # Mapas fixos para as categorias
    mappings = {
        "Status":  {"good": 1, "bad": 0},
        "Home":    {"rent": 0, "owner": 1, "parents": 2, "priv": 3, "other": 4, "ignore": 5},
        "Marital": {"married": 0, "widow": 1, "single": 2, "separated": 3, "divorced": 4},
        "Records": {"no": 0, "yes": 1},
        "Job":     {"freelance": 0, "fixed": 1, "partime": 2, "others": 3}
    }

    # Aplicar os mapeamentos
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Inferir tipos e converter inteiros para float64
    df = df.infer_objects(copy=False)
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = df[col].astype("float64")

    
    #for col in df.select_dtypes(include=["object"]).columns:
    #    df[col] = df[col].astype(str)
    #    df[col] = LabelEncoder().fit_transform(df[col])

    # Preencher valores nulos com 0 (caso algum valor categórico não tenha mapeamento)
    df.fillna(0, inplace=True)

    # Separar features e target
    X = df.drop(columns=["Status"])
    y = df["Status"].astype("int64")

    return X, y

#Main

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

dataset_original_raw = config["files"]["dataset"]
dataset_inferencias = config["files"]["inferences_log"]
report_datadrift_html = config["files"]["datadrift_html"]
report_datadrift_json = config["files"]["datadrift_json"]
report_classdrift_html =  config["files"]["classdrift_html"]
report_classdrift_json =  config["files"]["classdrift_json"]

df_original_dados, status = preprocess_data(pd.read_csv(dataset_original_raw))

df_inferencias = pd.read_csv(dataset_inferencias)
df_inferencias_dados = df_inferencias.drop(columns=['prediction', 'target'])

logging.info("Iniciando monitoramento de drifts")
logging.info(f"Dataset de treinamento: {dataset_original_raw}")
logging.info(f"Dataset de inferencias: {dataset_inferencias}")


## DATA DRIFT
#print(df_original_dados.head())
#print(df_inferencias_dados.head())
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_original_dados, current_data=df_inferencias_dados)
report.save_html(report_datadrift_html)
report.save_json(report_datadrift_json)
logging.info(f"Salvo relatórios de DataDrift: {report_datadrift_html} , {report_datadrift_json}")

## CLASS DRIFT
df_original_target = df_original_dados
df_original_target['target'] = status
df_original_target['prediction'] = status
#print(df_original_dados.head())
#print(df_inferencias.head())
report = Report(metrics=[ClassificationPreset()])
report.run(reference_data=df_original_target, current_data=df_inferencias)
report.save_html(report_classdrift_html)
report.save_json(report_classdrift_json)
logging.info(f"Salvo relatórios de ClassDrift: {report_classdrift_html}, {report_classdrift_json}")
