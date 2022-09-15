import os
import sys

path_manipulate_data = os.path.dirname(os.path.realpath(__file__))

class ManipulateData():
    def __init__(self) -> None:
        pass

    def get_path_raw_data(self):
        path_raw_data = \
            os.path.abspath(os.path.join(path_manipulate_data,
                            "..",
                            "1_raw_data"))
        
        return path_raw_data
    
    def get_path_exploratory_data(self):
        path_exploratory_data = \
            os.path.abspath(os.path.join(path_manipulate_data,
                            "..",
                            "2_exploratory_analysis",
                            "1_data"))
        return path_exploratory_data
    
    def get_path_output_format_data(self):
        path_exploratory_data = \
            self.get_path_exploratory_data()
        path_output_format_data = \
            os.path.join(path_exploratory_data, "pdm_exploratory.csv")
        return path_output_format_data

    def get_features_name(self):
        features_name = ["unit_number", "time", "setting_1", "setting_2",
                         "setting_3","sensor_1", "sensor_2", "sensor_3",
                         "sensor_4", "sensor_5", "sensor_6", "sensor_7",
                         "sensor_8", "sensor_9", "sensor_10", "sensor_11",
                         "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                         "sensor_16", "sensor_17", "sensor_18", "sensor_19",
                         "sensor_20", "sensor_21"]
        return features_name
