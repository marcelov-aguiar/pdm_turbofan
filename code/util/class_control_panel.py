import pandas as pd
from scipy.signal import savgol_filter
from typing import List, Tuple

class ControlPanel():
    """É responsável por definir parâmetros para fazer diferentes
    experimentos
    """
    def __init__(self,
                 rolling_mean: bool = False,
                 window_mean: int = 0,
                 is_grid_search: bool = False,
                 use_validation_data: bool = False,
                 use_optuna: bool = False,
                 number_units_validation: int = 0,
                 use_savgol_filter: bool = False) -> None:
        """É responsável por definir parâmetros para fazer diferentes
        experimentos.

        Parameters
        ----------
        rolling_mean : bool, optional.
            Caso True, será feita a média móvel dos dados conforme o número da
            unidade, by deafult False.
        window_mean : int, optional, optional.
            Tamanho da janela que será feita a média móvel, by default 0
        is_grid_search : bool, optional.
            Caso True, será feito GridSearch com validação cruzada, by default False.
        use_validation_data : bool, optional.
            Caso True, parte dos dados de treino é movido para os dados de teste,
            by default False
        use_optuna : bool, optional.
            Caso True, é usado o Optuna para otimizar os hiperparâmetros,
            by default False
        number_units_validation : int, optional
            Número de unidades que será migrado dos dados de treino para os
            dados de teste.
        apply_use_savgol_filter: bool, optional
            Caso True, filtro Savitzky–Golay é aplicado para diminuir o ruído dos
            dados, by default False
        """
        self.rolling_mean = rolling_mean
        self.window_mean = window_mean
        self.is_grid_search = is_grid_search
        self.use_validation_data = use_validation_data
        self.use_optuna = use_optuna
        self.number_units_validation = number_units_validation
        self.use_savgol_filter = use_savgol_filter

    def apply_use_savgol_filter(self,
                                df_data: pd.DataFrame,
                                ignore_column: List[str],
                                window_length: int = 15,
                                polyorder: int = 2) -> Tuple[pd.DataFrame, List[str]]:
        new_columns = []
        if self.use_savgol_filter:
            
            for column in df_data.columns:
                if column not in ignore_column:
                    df_data[f'{column}_savgol'] = savgol_filter(df_data[column],
                                                                window_length,
                                                                polyorder)
                    new_columns.append(f'{column}_savgol')
        return df_data, new_columns
    
    def apply_use_validation_data(self,
                                  df_train: pd.DataFrame,
                                  df_test: pd.DataFrame,
                                  column_name: str = 'unit_number') \
                                    -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.use_validation_data:
            units_quantity = self.number_units_validation
            units_numbers = df_train[column_name].unique()[-units_quantity:]
            for unit_number in units_numbers:
                df_aux = df_train[df_train[column_name] == unit_number].copy()
                df_train = df_train[~(df_train[column_name] == unit_number)]
                df_aux[column_name] = df_aux[column_name] + 100
                df_test = pd.concat([df_test, df_aux], axis=0)
            return df_train, df_test