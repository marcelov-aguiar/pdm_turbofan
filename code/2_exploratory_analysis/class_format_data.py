import pandas as pd
import numpy as np
import os
import sys

path_manipulate_data = os.path.join(sys.path[0],
                                    "..",
                                    "0_utils")
sys.path.append(path_manipulate_data)

from class_manipulate_data import ManipulateData


class FormatData(ManipulateData):
    def __init__(self):
        super().__init__()

    def read_txt_rul(self,
                     input_file_name: str) -> pd.DataFrame:
        path_raw_data = self.get_path_raw_data()

        path_data = os.path.join(path_raw_data, input_file_name)

        data = np.loadtxt(path_data)

        df_data = pd.DataFrame(data, columns=['max'])

        features_name = self.get_features_name()
        df_data[features_name[0]] = df_data.index.values + 1

        return df_data

    def format_raw_data(self, input_file_name: str) -> None:
        path_raw_data = self.get_path_raw_data()

        path_data = os.path.join(path_raw_data, input_file_name)

        data = np.loadtxt(path_data)

        features_name = self.get_features_name()

        df_data = pd.DataFrame(data, columns=features_name)

        get_path_exploratory_data = self.get_path_exploratory_data()

        input_file_name = os.path.splitext(input_file_name)[0]
        output_file_name = f"{input_file_name}_format.csv"

        path_output_format_data = os.path.join(get_path_exploratory_data,
                                               output_file_name)

        df_data.to_csv(path_output_format_data, index=False)

    def get_format_data(self, output_file_name: str) -> pd.DataFrame:
        get_path_exploratory_data = self.get_path_exploratory_data()

        output_file_name = os.path.splitext(output_file_name)[0]
        try:
            df_data = pd.read_csv(
                os.path.join(get_path_exploratory_data,
                             f"{output_file_name}_format.csv"))

        except Exception as e:
            print("Dado não encontrado. Execute método " +
                  f"format_raw_data para {output_file_name}.")
            print(e)

        return df_data
