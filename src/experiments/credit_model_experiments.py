import pandas as pd
import json
import logging 

import mlflow
import mlflow.data
from mlflow.models.signature import infer_signature

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


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

    # Preencher valores nulos com 0 (caso algum valor categórico não tenha mapeamento)
    df.fillna(0.0, inplace=True)

    # Inferir tipos e converter inteiros para float64
    df = df.infer_objects(copy=False)
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = df[col].astype("float64")

    # inteiro para evitar warning
    #df["Status"] = df["Status"].astype("int")
    
    # Separar features e target
    X = df.drop(columns=["Status"])
    y = df["Status"]
    

    logging.info('Estrutura do dataset:')
    for column in df.columns:
        logging.info(f"  Coluna: {column}, Tipo: {df[column].dtype}")

    # Criar dataset para MLflow
    mlflow_dataset = mlflow.data.from_pandas(df, targets="Status")

    return X, y, mlflow_dataset

def load_data(dataset_name):
    df = pd.read_csv(dataset_name)
    return df

def evaluate_and_log_model(model, model_name, params, X_train, X_test, y_train, y_test, mlflow_dataset, dataset_name):
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)

    with mlflow.start_run(run_name=f"{model_name}_{params}"):
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.set_tag("dataset_used", dataset_name)
        mlflow.set_tag("model_type", model_name)

        signature = infer_signature(X_train, y_test_pred)
        model_info = mlflow.sklearn.log_model(model, f"{model_name}_model", 
                                              signature=signature,
                                              input_example=X_train,
                                              registered_model_name=f"CreditCard_{model_name}")

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)
        result = pd.DataFrame(X_test, columns=X_train.columns)
        result["label"] = y_test.values
        result["predictions"] = predictions

        # evitar warning
        #result["label"] = result["label"].astype("int")

        mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="classifier",
        )

        logging.info(f"Modelo: {model_name}, Parâmetros: {params}")
        #print(result.head())

def run_all_models(X, y, mlflow_dataset, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg_grid = {
        "C": [0.01, 0.1, 1.0, 2.0, 3.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    for params in (dict(zip(logreg_grid.keys(), values)) for values in 
                   [(c, p, s) for c in logreg_grid["C"]
                              for p in logreg_grid["penalty"]
                              for s in logreg_grid["solver"]]):
        evaluate_and_log_model(logreg, "LogisticRegression", params, X_train, X_test, y_train, y_test, mlflow_dataset, dataset_name)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }

    for params in (dict(zip(rf_grid.keys(), values)) for values in 
                   [(n, d, s) for n in rf_grid["n_estimators"]
                              for d in rf_grid["max_depth"]
                              for s in rf_grid["min_samples_split"]]):
        evaluate_and_log_model(rf, "RandomForest", params, X_train, X_test, y_train, y_test, mlflow_dataset, dataset_name)

    # XGBoost
    #xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    xgb_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [6, 9, 12],
        "learning_rate": [0.1, 0.2, 0.3]
    }

    for params in (dict(zip(xgb_grid.keys(), values)) for values in 
                   [(n, d, l) for n in xgb_grid["n_estimators"]
                              for d in xgb_grid["max_depth"]
                              for l in xgb_grid["learning_rate"]]):
        evaluate_and_log_model(xgb, "XGBoost", params, X_train, X_test, y_train, y_test, mlflow_dataset, dataset_name)

def main():
    
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

    mlflow.set_tracking_uri(config["mlflow"]["mlflow_uri"])
    mlflow.set_experiment(config["mlflow"]["mlflow_experiment"])

    dataset_file = config["files"]["dataset"]
    df = load_data(dataset_file)
    
    logging.info(f"Pre-processamento do dataset {dataset_file}")
    
    X, y, mlflow_dataset = preprocess_data(df)
    
    logging.info("Iniciando treinameto dos modelos")

    run_all_models(X, y, mlflow_dataset, dataset_file)

if __name__ == "__main__":
    main()
