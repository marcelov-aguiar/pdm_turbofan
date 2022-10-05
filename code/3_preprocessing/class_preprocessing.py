import os
import sys

path_manipulate_data = os.path.join(sys.path[0],
                                    "..",
                                    "0_utils")
sys.path.append(path_manipulate_data)

from class_format_data import FormatData


class Preprocessing(FormatData):
    def __init__(self,
                 equipment_name: str) -> None:
        """_summary_

        Parameters
        ----------
        equipment_name : str
            Nome do equipamento que será executado o pré-processamento.
        """
        self.equipment_name = equipment_name

    def run_preprocessing(self):
        pass

    def training_data_preprocessing(self):
        pass

    def test_data_preprocessing(self):
        pass
