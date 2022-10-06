import sys
from pathlib import Path
import pandas as pd

path_explora_analysis = \
    Path(__file__).parent.parent.joinpath("2_exploratory_analysis")
sys.path.append(str(path_explora_analysis))


from class_format_data import FormatData


class Preprocessing(FormatData):
    def __init__(self,
                 equipment_name: str) -> None:
        """_summary_

        Parameters
        ----------
        equipment_name : str
            Nome do equipamento que será executado o pré-processamento.
            Pode assumir os seguintes valores: FD001, FD002, FD003 ou FD004.
        """
        super().__init__()
        self.equipment_name = equipment_name

    def run_preprocessing(self):
        self.training_data_preprocessing()

    def training_data_preprocessing(self):
        """Responsável por executar todas etapas do processamento
        para os dados de treino.
        """
        self.format_raw_data(f"train_{self.equipment_name}.txt")
        df_train = self.get_format_data(f"train_{self.equipment_name}.txt")
        df_train = self.remove_feat_low_variance(df_train)
        df_train = \
            df_train.groupby('unit_number').apply(self.add_rul).reset_index()
        df_train = df_train.drop(columns=["level_1", "index"])

        # salvando dados da base de dados de treino
        path_preprocessing_output = self.get_path_preprocessing_output()

        path_aux = path_preprocessing_output.joinpath(
            f"train_{self.equipment_name}.csv")

        df_train.to_csv(path_aux, index=False)

    def test_data_preprocessing(self):
        pass

    def remove_feat_low_variance(self,
                                 df_data: pd.DataFrame,
                                 VAR: float = 0.00000001) -> pd.DataFrame:
        """Responsável por remover as features de `df_data` com
        baixa variância.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame com as features a serem removidas.
        VAR : float, optional
            Valor da variância considerado como baixo. Abaixo desse valor
            as variáveis (features) serão removidas, by default 0.00000001.

        Returns
        -------
        pd.DataFrame
            DataFrame sem as features com baixa variância.
        """
        good_sensor = list(df_data.columns)
        for sensor in df_data.columns:
            if df_data[sensor].var() <= VAR:
                print("Sensor is flat:", sensor)
                good_sensor.remove(sensor)
        df_data = df_data[good_sensor]
        return df_data

    def add_rul(self,
                df_unit: pd.DataFrame) -> pd.DataFrame:
        """Responsável por criar a feature RUL (Remaining
        Useful Life) para cada unidade do DataFrame principal.

        Parameters
        ----------
        df_unit : pd.DataFrame
            DataFrame de uma unidade (parte do DataFrame principal).

        Returns
        -------
        pd.DataFrame
            DataFrame da unidade com RUL calculado.
        """
        df_unit['RUL'] = [max(df_unit['time'])] * len(df_unit)
        df_unit['RUL'] = df_unit['RUL'] - df_unit['time']
        del df_unit['unit_number']
        return df_unit.reset_index()


if __name__ == '__main__':
    preprocessing = Preprocessing('FD001')
    preprocessing.run_preprocessing()
    print("Teste")
