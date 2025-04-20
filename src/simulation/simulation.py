import requests
import json
import logging

import random
import pandas as pd
from time import sleep
import numpy as np

def random_normal_fload(min_val, max_val, media, desvio):
    valor = random.gauss(media, desvio)
    valor = max(min_val, min(max_val, valor))
    return round(valor,1)

def random_normal_integer(min_val, max_val, media, desvio):
    valor = random.gauss(media, desvio)
    valor = max(min_val, min(max_val, valor))
    return int(valor)

# Função para gerar uma linha de dados sintéticos
def gerar_dado():
    return {
        "Seniority":  random_normal_integer(0, 48, 5, 8),
        "Home": random.randint(0, 5),
        "Time": random_normal_integer(1, 72, 48, 14),
        "Age": random_normal_integer(18,80,36,10), #random.randint(18, 80),
        "Marital": random.randint(0, 4),
        "Records": random.randint(0, 1),
        "Job": random.randint(0, 3),
        "Expenses": random_normal_fload(35, 180, 51, 19),
        "Income": random_normal_fload(6, 959, 125, 80),
        "Assets": random_normal_fload(0, 300000, 3000, 11574),
        "Debt": random_normal_fload(0, 30000, 0, 1245),
        "Amount": random_normal_fload(100, 5000, 1000, 474),
        "Price": random_normal_fload(100, 12000, 1400, 628)
    }


def servidor_ativo(url):
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"dataframe_records": [{}]}  # Payload mínimo, só pra testar
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        #print(f"Status Code: {response.status_code}")
        
        # Considere 200 ou 400 como "servidor ativo"
        if response.status_code in (200, 400):
            return True
        else:
            #print(f"Servidor respondeu com código {response.status_code}.")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro na requisição: {e}")
        return False
    
def predicoes_api(num_amostras):
    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}

    # Lista para armazenar os dados com predição
    dados_com_predicao = []

    for i in range(num_amostras):
        dado = gerar_dado()
        payload = {"dataframe_records": [dado]}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            pred = response.json()
            dado["prediction"] = int(pred['predictions'][0])
    
        except Exception as e:
            print(f"[{i+1}/100] Erro na requisição:", e)
            dado["prediction"] = None  # fallback
            dado["target"] = None 

        #print (dado, "\n")
        dados_com_predicao.append(dado)
        #sleep(0.1)

    return dados_com_predicao

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
logging.info("Iniciando uma simulação usando a API para criar uma dataset de inferências.")

if not servidor_ativo("http://127.0.0.1:8000/invocations"):
        logging.info("Servidor http://127.0.0.1:8000/invocations não está disponível. Parando simulação.")
        exit()

dataset_inferencias = config["files"]["inferences_log"]
acuracia_dados_simulacao = config["simulation"]["acuracia"]
num_amostras_simulacao = config["simulation"]["amostras"]

df = pd.DataFrame(predicoes_api(num_amostras_simulacao))

# simula acurácia dos dados criando os targets
mask = np.random.rand(len(df)) < acuracia_dados_simulacao
df['target'] = np.where(mask, df['prediction'], 1 - df['prediction'])

logging.info(f"Dataset terá {num_amostras_simulacao} amostras e acurácia de {acuracia_dados_simulacao}")

df.to_csv(dataset_inferencias, index=False)
logging.info(f"Dataset de simulação da API salvo em: {dataset_inferencias}")

