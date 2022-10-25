import sys
import pandas as pd
import numpy as np
from typing import List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import logging
from pathlib import Path


# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion

path_manipulate_data = Path(__file__).parent.parent.parent.joinpath("0_utils")

sys.path.append(str(path_manipulate_data))

from class_manipulate_data import ManipulateData

manipulate_data = ManipulateData()
path_preprocessing_output = manipulate_data.get_path_preprocessing_output()


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Faz o cálculo do RMSE.

    Parameters
    ----------
    y_true : pd.Series
        Valor real.
    y_pred : pd.Series
        Valor predito pelo modelo.

    Returns
    -------
    float
        Valor do RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_scatter_performance_individual(y_true: np.array,
                                        y_pred: np.array,
                                        name_output: str,
                                        subtitle: str,
                                        df_metrics: pd.DataFrame = None) \
                                            -> Figure:
    """Retorna a figura do gráfico `regplot` do `seaborn` com o valor predito
    e o valor real. Além dissso, acrescenta no gráfico os dados da métrica que
    estão no `df_df_metrics`.

    Parameters
    ----------
    y_true : np.array
        Array com os valores reais.
    y_pred : np.array
        Array com os valores preditos.
    name_output : str
        Nome da feature predita.
    subtitle : str
        Legenda que será mostrada no gráfico.
    df_metrics : pd.DataFrame, optional
        DataFrame de uma única linha. Onde as colunas contém
        o nome das métricas e a linha os valores de cada uma, by default None

    Returns
    -------
    Figure
        Gráfico no formato de figura do matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    max_value = max([y_true.max(), y_pred.max()]) + 0.05
    min_value = min([y_true.min(), y_pred.min()]) - 0.05
    identity_line = np.linspace(min_value, max_value)

    ax.plot(identity_line, identity_line, '-', color='darkgrey', linewidth=3)
    sns.regplot(x=y_true, y=y_pred,
                color='darkcyan',
                fit_reg=True,
                ax=ax,
                truncate=False)

    ax.set_ylim((min_value, max_value))
    ax.set_xlim((min_value, max_value))
    ax.set_ylabel(f'{name_output} predito', fontsize=16)
    ax.set_xlabel(f'{name_output} medido', fontsize=16)
    ax.set_title(f'Actual X Pred - {subtitle}', fontsize=18)

    if df_metrics is not None:
        test_text = df_metrics.iloc[0].to_string(float_format='{:,.2f}'.format)
        ax.text(0.1, 0.8, test_text, color='indigo',
                bbox=dict(facecolor='none', edgecolor='indigo',
                          boxstyle='round,pad=1'),
                transform=ax.transAxes, fontsize=9)

    return fig


def create_df_metrics(true: pd.Series,
                      pred: pd.Series) -> pd.DataFrame:
    """Reponsável por calcular as métricas do modelo e
    retornar um DataFrame com todas as métricas calculadas.

    Parameters
    ----------
    y_true : pd.Series
        Valor real.
    y_pred : pd.Series
        Valor predito.

    Returns
    -------
    pd.DataFrame
        DataFrame com as métricas.
    """
    metrics = {}
    metrics['MAE'] = mean_absolute_error(true, pred)
    metrics['RMSE'] = root_mean_squared_error(true, pred)
    metrics['R2'] = r2_score(true, pred)

    df_metrics = pd.DataFrame(metrics, index=['Value'])

    return df_metrics


def plot_prediction(name_output: pd.DataFrame,
                    y_real: pd.Series,
                    y_pred: pd.Series,
                    title: str = 'Turbofan') -> Figure:
    """Retorna uma figura de um gráfico de predição. Onde
    o eixo x é do tipo timestamp, o eixo contém as duas feature,
    sendo elas: Real e a predita.

    Parameters
    ----------
    df_plot : pd.DataFrame
        DataFrame com a feature real e a feature predita no
        forma: `NOME_PRED`.
    df_periods : pd.DataFrame
        DataFrame de operação do forno.
    output : str
        Nome da feature a ser plotada.
    Returns
    -------
    Figure
        Figura com o gráfico da feature predita e real.
    """
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=(25, 6.7))
    ax.set_title(f'Prediction Real x Pred - {title}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel(f'{name_output}')

    ax.plot(y_real,
            'o', color='royalblue', label='Real')

    ax.plot(y_pred,
            'o--', color='firebrick', label='Predicted', linewidth=2)

    ax.legend(ncol=2)

    return fig


def plot_features_importance(features_name: List[str],
                             coef_features: np.ndarray) -> Figure:
    """Responsável por criar um gráfico com o nome e o peso das features.

    Parameters
    ----------
    features_name : List[str]
        Nome das features.
    coef_features : np.array
        Coeficientes das features.

    Returns
    -------
    Figure
        Gráfico com o nome e o peso das features
    """
    df_coef = pd.DataFrame(columns=["Feature", "Coef"])
    df_coef["Feature"] = features_name
    df_coef["Coef"] = coef_features

    df_coef = df_coef[df_coef["Coef"] != 0]
    df_coef.sort_values("Coef", inplace=True, ascending=True)
    plt.rcParams['font.size'] = 11
    fig = plt.figure(figsize=(15, 9))
    plt.barh(df_coef["Feature"], df_coef["Coef"],
             color='darkblue')

    plt.barh(df_coef["Feature"][df_coef["Coef"] < 0],
             df_coef["Coef"][df_coef["Coef"] < 0],
             color='maroon')

    plt.title('Coeficientes Modelo')
    plt.yticks(rotation=45)

    return fig


def save_metrics_mlflow(df_metrics: pd.DataFrame,
                        metric_legend: str):
    """Salva as métricas no MLFlow com base no DataFrame
    `df_metrics` com as métricas e com a legenda que identifica
    o tipo de métrica (train ou test).

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame de uma única linha que contém nas colunas o nome
        das métricas calculadas.
    metric_legend : str
        Nome da legenda da métrica salva no MLFlow. Pode assumir valor
        de train ou test.
    """
    for metric_name in df_metrics.columns:
        mlflow.log_metric(f'{metric_legend}_{metric_name}',
                          df_metrics[metric_name][0])


class ControlPanel():
    """É responsável por definir parâmetros para fazer diferentes
    experimentos
    """
    def __init__(self,
                 rolling_mean: bool = False,
                 window_mean: int = 0,
                 is_grid_search: bool = False) -> None:
        """É responsável por definir parâmetros para fazer diferentes
        experimentos.

        Parameters
        ----------
        rolling_mean : bool, optional.
            Caso True, será feita a média móvel dos dados conforme o número da
            unidade, by deafult False.
        window_mean : int, optional.
            Tamanho da janela que será feita a média móvel, by default 0
        rolling_mean : bool
            Caso True, será feito GridSearch com validação cruzada, by default False.
        """
        self.rolling_mean = rolling_mean
        self.window_mean = window_mean
        self.is_grid_search = is_grid_search


logger.info("Definindo as entradas, a saída e o equipamento.")
# features selecionadas pela variância
# input_model = ['setting_1', 'setting_2', 'sensor_2', 'sensor_3',
#                'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
#                'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
#                'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# # todas as entradas
input_model = [
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11',
    'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
    'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

output_model = ['RUL']

equipment_name = 'FD001'

control_panel = ControlPanel(rolling_mean=False,
                             window_mean=24,
                             is_grid_search=True)

logger.info("Lendo os dados de treino.")

path_dataset_train = \
    str(path_preprocessing_output.joinpath(f"train_{equipment_name}.csv"))

df_train = pd.read_csv(path_dataset_train)

logger.info("Lendo os dados de teste.")

path_dataset_test = \
    str(path_preprocessing_output.joinpath(f"test_{equipment_name}.csv"))

df_test = pd.read_csv(path_dataset_test)

logger.info("Criando o modelo.")
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('FD001')
with mlflow.start_run(run_name='RandomForest'):
    model = RandomForestRegressor()
    pipeline = Pipeline([('std', StandardScaler()), ('regressor', model)])

    pipeline = TransformedTargetRegressor(regressor=pipeline,
                                       transformer=StandardScaler())
    if control_panel.is_grid_search:
        param_grid = {
            "regressor__regressor__max_depth": [4, 5, 6, 7, 8, 9, 10],
            "regressor__regressor__max_features": [2, 3, 4, 5],
            "regressor__regressor__n_estimators": [10, 100, 200],
            "regressor__regressor__min_samples_split": [2, 4, 8, 10]
        }
        model = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                             cv=3, n_jobs=-1, verbose=2)
    else:
        model = pipeline

    if control_panel.rolling_mean:
        df_rolling = \
            df_train.groupby('unit_number').rolling(
                window=control_panel.window_mean).mean()

        df_rolling = df_rolling.dropna()
        df_rolling = df_rolling.reset_index()
        df_train = df_rolling.copy()

    y_train = df_train[output_model]
    X_train = df_train[input_model]

    logger.info("Treinando o modelo.")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    df_metrics_train = create_df_metrics(y_train, y_train_pred)
    logger.info("Salvando as métricas de treino no MLFlow.")
    save_metrics_mlflow(df_metrics_train, 'train')

    fig = plot_scatter_performance_individual(y_train.values,
                                              y_train_pred,
                                              'RUL',
                                              'Train',
                                              df_metrics_train)
    mlflow.log_figure(fig, artifact_file='plot_scatter_train.png')

    fig = plot_prediction(output_model[0], y_train.values, y_train_pred)
    mlflow.log_figure(fig, artifact_file='plot_pred_train.png')

    if control_panel.rolling_mean:
        df_rolling = \
            df_test.groupby('unit_number').rolling(
                window=control_panel.window_mean).mean()
        df_rolling = df_rolling.dropna()
        df_rolling = df_rolling.reset_index()
        df_test = df_rolling.copy()
    y_test = df_test[output_model]
    X_test = df_test[input_model]

    y_test_pred = model.predict(X_test)
    df_metrics_test = create_df_metrics(y_test, y_test_pred)
    logger.info("Salvando as métricas de teste no MLFlow.")
    save_metrics_mlflow(df_metrics_test, 'test')

    fig = plot_scatter_performance_individual(y_test.values,
                                              y_test_pred,
                                              'RUL',
                                              'Test',
                                              df_metrics_test)
    mlflow.log_figure(fig, artifact_file='plot_scatter_test.png')

    fig = plot_prediction(output_model[0], y_test.values, y_test_pred)
    mlflow.log_figure(fig, artifact_file='plot_pred_test.png')

    logger.info("Salvando artefatos no MLFlow.")
    info_columns = []
    for column in input_model:
        info_columns.append(ColSpec(('double'), column))
    input_schema = Schema(info_columns)

    output_schema = Schema([ColSpec(("double"), output_model[0])])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.sklearn.log_model(model,
                             'TURBOFAN',
                             signature=signature)

    mlflow.log_param('regressor', pipeline)
    mlflow.log_param('control panel', control_panel.__dict__)

    logger.info("Salvando coeficientes do modelo.")
    if control_panel.is_grid_search:
        fig = plot_features_importance(input_model,
                                       model.best_estimator_.regressor_['regressor']
                                       .feature_importances_)
        mlflow.log_param('best_params', model.best_params_)
    else:
        fig = plot_features_importance(input_model,
                                       model.regressor_['regressor']
                                       .feature_importances_)

    mlflow.log_figure(fig, artifact_file='feature_importance.png')
