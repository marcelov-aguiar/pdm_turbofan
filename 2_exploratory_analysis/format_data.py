import pandas as pd
import numpy as np
import os
import sys

path_manipulate_data = os.path.join(sys.path[0],
                                    "..",
                                    "0_utils")
sys.path.append(path_manipulate_data)

from class_manipulate_data import ManipulateData

manipulate_data = ManipulateData()

file_name = 'train_FD001.txt'

path_raw_data = manipulate_data.get_path_raw_data()

path_data_train = os.path.join(path_raw_data,file_name)


data_train = np.loadtxt(path_data_train)

features_name = manipulate_data.get_features_name()

df_train = pd.DataFrame(data_train, columns=features_name)

path_output_format_data = manipulate_data.get_path_output_format_data()

df_train.to_csv(path_output_format_data, index=False)
