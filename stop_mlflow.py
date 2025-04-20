import subprocess

def stop_mlflow():
    # Lista processos com 'mlflow' no comando
    result = subprocess.run('wmic process where "commandline like \'%mlflow%\'" get ProcessId', 
                            capture_output=True, text=True, shell=True)
    pids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip().isdigit()]
    
    if pids:
        for pid in pids:
            subprocess.run(f'taskkill /F /PID {pid}', shell=True)
        print(f"Finalizado(s) processo(s) MLflow: {', '.join(pids)}")
    else:
        print("Nenhum processo MLflow encontrado.")

stop_mlflow()
