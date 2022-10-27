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
                 number_units_validation: int = 0) -> None:
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
        """
        self.rolling_mean = rolling_mean
        self.window_mean = window_mean
        self.is_grid_search = is_grid_search
        self.use_validation_data = use_validation_data
        self.use_optuna = use_optuna
        self.number_units_validation = number_units_validation
