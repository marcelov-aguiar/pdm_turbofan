from pathlib import Path
from typing import List

path_manipulate_data = Path(__file__).parent


class ManipulateData():
    """Responsável por guardar informações da localização dos
    repositórios e dos dados específicos do projeto.
    """
    def __init__(self) -> None:
        pass

    def get_path_raw_data(self) -> Path:
        """Responsável por retornar o path onde fica
        localizado os dados brutos dos equipamentos do
        projeto.

        Returns
        -------
        Path
            path onde fica localizado os dados brutos dos
            equipamentos do projeto.
        """
        path_raw_data = path_manipulate_data.parent.joinpath('1_raw_data')

        return path_raw_data

    def get_path_exploratory_data(self) -> Path:
        """Responsável por retornar o path onde fica
        localizados os dados analisados na etapa de
        análise exploratória.

        Returns
        -------
        Path
            path onde fica localizados os dados analisados
            na etapa de análise exploratória.
        """
        path_exploratory_data = \
            path_manipulate_data.parent.joinpath('2_exploratory_analysis',
                                                 '1_data')

        return path_exploratory_data

    def get_path_preprocessing_output(self) -> Path:
        """Responsável por retornar o path onde fica localizado
        os dados de saída do pré-processamento.

        Returns
        -------
        Path
            path onde fica localizado os dados de saída do
            pré-processamento.
        """
        path_preprocessing_output = \
            path_manipulate_data.parent.joinpath('3_preprocessing',
                                                 'out')

        return path_preprocessing_output

    def get_features_name(self) -> List[str]:
        """Responsável por retornar o nome das colunas dos
        dados brutos (localizados em `path_raw_data`).

        Returns
        -------
        List[str]
            Lista com nome das colunas dos dados brutos
            (localizados em `path_raw_data`)
        """
        features_name = ["unit_number", "time", "setting_1", "setting_2",
                         "setting_3", "sensor_1", "sensor_2", "sensor_3",
                         "sensor_4", "sensor_5", "sensor_6", "sensor_7",
                         "sensor_8", "sensor_9", "sensor_10", "sensor_11",
                         "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                         "sensor_16", "sensor_17", "sensor_18", "sensor_19",
                         "sensor_20", "sensor_21"]
        return features_name
