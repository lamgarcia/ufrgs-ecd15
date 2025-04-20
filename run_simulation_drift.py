import subprocess
import os
import time

def run_command(cmd, shell=True, background=False):
    if background:
        subprocess.Popen(cmd, shell=shell)
    else:
        subprocess.run(cmd, shell=shell, check=True)

print("Simulando o uso da api...")
run_command("python src\\simulation\\simulation.py")

print("Gerando relatório de monitoramento de drifts...")
run_command("python src\\monitor\\monitor_drift.py")

print("Verifica se aciona trigger de retreinamento do modelo...")
run_command("python src\\triggers\\retraining_trigger.py")