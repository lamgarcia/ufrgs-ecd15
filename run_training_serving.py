import subprocess
import os
import time
import socket

def run_command(cmd, shell=True, background=False):
    if background:
        subprocess.Popen(cmd, shell=shell)
    else:
        subprocess.run(cmd, shell=shell, check=True)

def is_mlflow_running():
    result = subprocess.run('tasklist', capture_output=True, text=True)
    return 'mlflow.exe' in result.stdout or 'python.exe' in result.stdout and 'mlflow' in result.stdout

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if is_port_in_use(5000):
    print("MLflow UI já está em execução na porta 5000. ")
else:
    #print("Criando variável de ambiente MLFLOW_TRACKING_URI para ambiente WINDOWS.")
    os.system('setx MLFLOW_TRACKING_URI "sqlite:///mlflow.db"')

    print("Iniciando MLflow UI em background...")
    run_command("start /B mlflow ui --backend-store-uri sqlite:///mlflow.db", background=True)
     
    # Aguarda um pouco para o MLflow UI subir
    print ("Aguardando o MLFlow subir...")
    time.sleep(7)

print("Treinando modelo (experiment)...")
run_command("python src\\experiments\\credit_model_experiments.py")

print("Promovendo melhor modelo de maior F1 score...")
run_command("python src\\experiments\\credit_model_promote.py")

print("Subindo API com modelo campeão...")
run_command("python src\\serve\\credit_model_serve.py", background=True)

