import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
import util
from class_control_panel import ControlPanel
from class_manipulate_data import ManipulateData
import class_genetic_selection as ga
from functools import partial

# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion

logger.info(util.init())

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

    df_coef["Feature"] = df_coef['Feature'].replace(' ',
                                                    '*',
                                                    regex=True)

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


logger.info("Definindo as entradas, a saída e o equipamento.")
# # features selecionadas pela variância
# input_model = ['setting_1', 'setting_2', 'sensor_2', 'sensor_3',
#                'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
#                'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
#                'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# todas as entradas
input_model = ['time',
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11',
    'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
    'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

output_model = ['RUL']

equipment_name = 'FD003'

control_panel = ControlPanel(rolling_mean=False,
                             window_mean=24,
                             use_validation_data=True,
                             number_units_validation=10,
                             use_savgol_filter=False)

logger.info("Lendo os dados de treino.")

path_dataset_train = \
    str(path_preprocessing_output.joinpath(f"train_{equipment_name}.csv"))

df_train = pd.read_csv(path_dataset_train)

logger.info("Lendo os dados de teste.")

path_dataset_test = \
    str(path_preprocessing_output.joinpath(f"test_{equipment_name}.csv"))

df_test = pd.read_csv(path_dataset_test)

df_train, df_test = control_panel.apply_use_validation_data(df_train,
                                                            df_test,
                                                            'unit_number')


def stop_condition(gen_number, population, AGE_LIMIT, MAX_GENERATIONS, MIN_GENERATIONS):
    return gen_number >= MAX_GENERATIONS or \
           (gen_number >= MIN_GENERATIONS and
            population.get_oldest_individual().get_age() >= AGE_LIMIT)


def apply_genetic(df_train,
                  df_test,
                  output_model,
                  exclude_columns=['unit_number']+output_model):
    # pipeline = Pipeline([("poly", PolynomialFeatures()),
    #                      ('std', StandardScaler())])
    # aplicar aos dados poly e scaler
    poly = PolynomialFeatures(2)
    sclin = StandardScaler()
    sclout = StandardScaler()

    X_train = df_train.drop(exclude_columns, axis=1)
    X_train = poly.fit_transform(X_train)
    X_train = sclin.fit_transform(X_train)
    X_train = pd.DataFrame(X_train,
                           columns=poly.get_feature_names(
                            df_train.drop(exclude_columns,
                                          axis=1).columns))

    X_test = df_test.drop(exclude_columns, axis=1)
    X_test = poly.transform(X_test)
    X_test = sclin.transform(X_test)
    X_test = pd.DataFrame(X_test,
                          columns=poly.get_feature_names(
                            df_train.drop(exclude_columns,
                                          axis=1).columns))
    
    Y_train = df_train[output_model]
    Y_train = sclout.fit_transform(Y_train)
    Y_train = pd.DataFrame(Y_train, columns=[output_model],
                           index=df_train.index)

    Y_test = df_test[output_model]
    Y_test = sclout.transform(Y_test)
    Y_test = pd.DataFrame(Y_test, columns=[output_model],
                          index=df_test.index)

    used_columns = list(X_train.columns)

    model_constructor = partial(LinearRegression, fit_intercept=False)

    individual_constructor = partial(ga.FeatureSelectionIndividual,
                                     model_constructor=model_constructor,
                                     possible_features=X_train.columns,
                                     past_scores_to_keep=10,
                                     extra_feature_penalty=0.0003,
                                     features_without_penalty=15)

    N_PARENTS = 80
    POPULATION_SIZE = 150
    MIN_GENERATIONS = 100
    MAX_GENERATIONS = 500
    AGE_LIMIT = 15
    USE_RECOMBINATION = True
    MUTATION_RATE = 0.05

    percentiles = [100, 75, 50, 25, 0]


    percentiles_track = pd.DataFrame(columns=list(map(str, percentiles)))
    percentiles_track.index.name = 'Generation'
    age_track = pd.DataFrame(columns=['Age'])
    age_track.index.name = 'Generation'

    population = ga.Population(individual_constructor,
                                POPULATION_SIZE,
                                N_PARENTS,
                                USE_RECOMBINATION,
                                MUTATION_RATE)
    population.set_data(X_train, Y_train)
    population.fit()

    g = 0
    while not stop_condition(g, population, AGE_LIMIT, MAX_GENERATIONS, MIN_GENERATIONS):
        print(f'\rProcessing generation {g + 1}', end='')

        population.select_parents()
        population.reproduce()
        population.mutate()
        population.join_generation()
        population.fit()
        g += 1

        generation_percentiles = \
            population.get_fitness_percentiles(percentiles)

        aux = pd.DataFrame(generation_percentiles,
                           columns=list(generation_percentiles.keys()),
                           index=[0])

        percentiles_track = \
            pd.concat([percentiles_track, aux]).reset_index().drop(columns=['index'])
        # percentiles_track = \
        #     percentiles_track.append(generation_percentiles,
        #                              ignore_index=True)
        # age_track = \
        #     age_track.append(
        #         {'Age':
        #          population.get_oldest_individual().get_age()},
        #         ignore_index=True)

    print('\rProcessing finished')

    best_individual = population.get_best_individual()
    oldest_individual = population.get_oldest_individual()
    model = oldest_individual.get_model()
    used_columns = oldest_individual.get_used_features()
    print(f'Best individual age: {best_individual.get_age()}')
    print(f'Oldest individual age: {oldest_individual.get_age()}')
    print(f'Selected {len(used_columns)} out of + '
          f'{len(X_train.columns)} possible features')
    print('Used features (with correspondent coefficient):')
    # list(zip(used_columns, model.coef_))
    print(used_columns)

apply_genetic(df_train, df_test, output_model)
