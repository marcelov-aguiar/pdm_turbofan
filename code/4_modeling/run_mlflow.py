import os

path_mlruns = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mlruns_2')

path_sqlite = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mlflow_2.db')

os.system(f'mlflow ui --backend-store-uri sqlite:///"{path_sqlite}" --default-artifact-root ./"{path_mlruns}" --host 127.0.0.1 --port 5000 &')
