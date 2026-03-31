import os
import torch
import datetime
import traceback
import numpy as np
import pandas as pd

from bus_routing.Map_graph import Map_graph
from bus_routing.Data_structures import Data_folders, Temporal_data_config
from bus_routing.generative_models.Requests_locations_module import Requests_locations_module
from bus_routing.generative_models.Requests_number_module import Requests_number_module
from bus_routing.generative_models.data_provider.Requests_locations_dataset import Request_Locations_Data_Manager, Requests_locations_dataset, Requests_locations_time_series
from bus_routing.generative_models.data_provider.Requests_number_dataset import Request_Number_Data_Manager, Requests_numbers_dataset, Requests_numbers_time_series

class Request_Number_Prediction_Manager:

    def __init__(self, data_folders: Data_folders, num_intervals: int, checkpoint_file_path: str,
                 load_predictions: bool, label_length: int):
        self.checkpoint_path = checkpoint_file_path
        self.num_intervals = num_intervals
        self.label_length = label_length
        self.data_folders = data_folders
        if load_predictions:
            self.prediction_df = self._load_request_predictions()
        else:
            self.prediction_df = self._initialize_prediction_df()
    
    def _initialize_prediction_df(self):
        interval_length = 60 // self.num_intervals
        temporal_config = Temporal_data_config(interval_length=interval_length)
        request_data_manager = Request_Number_Data_Manager(data_folders=self.data_folders,
                                                temporal_config=temporal_config,
                                                preprocess=False,
                                                save_data=False)
        
        train_data_df, train_slice_start_points_df = request_data_manager.get_requests_numbers_training_data()
        val_data_df, val_slice_start_points_df = request_data_manager.get_requests_numbers_validation_data()
        test_data_df, test_slice_start_points_df = request_data_manager.get_requests_numbers_testing_data()

        slice_start_points_dict = {"train": train_slice_start_points_df["start_points"].tolist(),
                                "test": test_slice_start_points_df["start_points"].tolist(),
                                "val": val_slice_start_points_df["start_points"].tolist()}

        time_series = Requests_numbers_time_series(train_data_df=train_data_df,
                                                   val_data_df=val_data_df,
                                                   test_data_df=test_data_df,
                                                   temporal_config=temporal_config)
        
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        train_prediction_df = self._generate_prediction_df(split="train",
                                                          time_series=time_series,
                                                          slice_start_points_dict=slice_start_points_dict,
                                                          data_df=train_data_df,
                                                          slice_start_points_df=train_slice_start_points_df,
                                                          device=device)
        
        val_prediction_df = self._generate_prediction_df(split="val",
                                                         time_series=time_series,
                                                         slice_start_points_dict=slice_start_points_dict,
                                                         data_df=val_data_df,
                                                         slice_start_points_df=val_slice_start_points_df,
                                                         device=device)
        
        test_prediction_df = self._generate_prediction_df(split="test",
                                                          time_series=time_series,
                                                          slice_start_points_dict=slice_start_points_dict,
                                                          data_df=test_data_df,
                                                          slice_start_points_df=test_slice_start_points_df,
                                                          device=device)

        prediction_df = pd.concat([train_prediction_df, val_prediction_df, test_prediction_df], axis=0, ignore_index=True)
        self._save_request_predictions(prediction_df=prediction_df)

        return prediction_df
    
    def _generate_prediction_df(self, split: str, time_series: Requests_locations_time_series,
                                slice_start_points_dict, data_df, slice_start_points_df, device):
        
        request_number_dataset = Requests_numbers_dataset(request_time_series=time_series,
                                                          slice_start_points_dict=slice_start_points_dict,
                                                          split=split,
                                                          sequence_length=self.num_intervals,
                                                          label_length=label_length,
                                                          prediction_length=self.num_intervals,)
        
        prediction_dict = self._generate_request_predictions(request_number_dataset=request_number_dataset,
                                                             test_data_df=data_df,
                                                             test_slice_start_points_df=slice_start_points_df,
                                                             device=device)
        prediction_df = pd.DataFrame(prediction_dict)

        return prediction_df

    
    def _load_model(self, device: torch.device):
        print("Loading request number predictor ...")
        requests_number_forecaster = Requests_number_module.load_from_checkpoint(checkpoint_path=self.checkpoint_path)
        requests_number_forecaster.to(device)
        print("Request number predictor has been loaded!")
        
        return requests_number_forecaster
    
    def _populate_prediction_dict(self, start_row, prediction_dict):
        prediction_dict["year"].append(int(start_row["year"]))
        prediction_dict["month"].append(int(start_row["month"]))
        prediction_dict["day"].append(int(start_row["day"]))
        prediction_dict["hour"].append(int(start_row["hour"]))
        prediction_dict["interval_index"].append(int(start_row["interval_index"]))

        print(f"Generating predictions for {prediction_dict["year"][-1]}-{prediction_dict["month"][-1]}-{prediction_dict["day"][-1]}-{prediction_dict["hour"][-1]}-{prediction_dict["interval_index"][-1]}")
    
    def _generate_request_predictions(self, request_number_dataset: Requests_numbers_dataset,
                                      test_data_df: pd.DataFrame, test_slice_start_points_df: pd.DataFrame,
                                      device: torch.device):
        prediction_dict = {"year": [],
                            "month": [],
                            "day": [],
                            "hour": [],
                            "interval_index": [],
                            "prediction": [],
                            "true_value": []}
        
        requests_number_forecaster = self._load_model(device=device)

        print("Generating predictions for number of requests ...")
        for i, start_point in enumerate(test_slice_start_points_df["start_points"]):
            start_row = test_data_df.iloc[start_point + request_number_dataset.sequence_length]
            self._populate_prediction_dict(start_row=start_row,
                                           prediction_dict=prediction_dict)

            seq_x, seq_y, seq_x_mark, seq_y_mark = request_number_dataset.__getitem__(i=i)

            seq_x = seq_x.unsqueeze(0)
            seq_y = seq_y.unsqueeze(0)
            seq_x_mark = seq_x_mark.unsqueeze(0)
            seq_y_mark = seq_y_mark.unsqueeze(0)

            true = seq_y[:, -requests_number_forecaster.model_config.pred_len:, :].numpy()
            shape = true.shape
            scaled_true = requests_number_forecaster.scaler.inverse_transform(true.reshape(shape[0] * shape[1], -1)).reshape(shape)
            scaled_true = scaled_true[:, :, -1:]

            pred = requests_number_forecaster.predict(x=seq_x, x_mark=seq_x_mark, y_mark=seq_y_mark)
            pred = pred.flatten()
            prediction_dict["prediction"].append(pred)
            prediction_dict["true_value"].append(scaled_true)
        
        print("Predicted number of requests has been generated!")

        return prediction_dict
    
    def _save_request_predictions(self, prediction_df: pd.DataFrame):
        print("Saving number predictions ...")
        prediction_df.to_pickle(os.path.join(self.data_folders.predicted_requests_folder_path, "request_numbers.pkl"))
        prediction_df.to_csv(os.path.join(self.data_folders.predicted_requests_folder_path, "request_numbers.csv"))
        print("Number predictions have been saved!")
    
    def _load_request_predictions(self):
        print("Loading number predictions ...")
        prediction_df = pd.read_pickle(os.path.join(self.data_folders.predicted_requests_folder_path, "request_numbers.pkl"))
        print("Number predictions have been loaded ...")
        return prediction_df


class Request_Locations_Prediction_Manager:

    def __init__(self, data_folders: Data_folders, num_intervals: int, checkpoint_file_path: str,
                 load_predictions: bool):
        self.num_intervals = num_intervals
        self.checkpoint_path = checkpoint_file_path
        self.data_folders = data_folders
        if load_predictions:
            self.prediction_df = self._load_request_predictions()
        else:
            self.prediction_df = self._initialize_prediction_df()

    def _initialize_prediction_df(self):
        interval_length = 60 // self.num_intervals
        temporal_config = Temporal_data_config(interval_length=interval_length)
        map_object = Map_graph(initialize_shortest_path=False, 
                               routing_data_folder=self.data_folders.routing_data_folder,
                               area_text_file=self.data_folders.area_text_file, 
                               use_saved_map=True, 
                               save_map_structure=False)
        
        request_data_manager = Request_Locations_Data_Manager(data_folders=self.data_folders,
                                                              temporal_config=temporal_config,
                                                              map_graph=map_object,
                                                              preprocess=False,
                                                              save_data=False)
        
        train_data_df, train_slice_start_points_df = request_data_manager.get_requests_locations_training_data()
        val_data_df, val_slice_start_points_df = request_data_manager.get_requests_locations_validation_data()
        test_data_df, test_slice_start_points_df = request_data_manager.get_requests_locations_testing_data()
        
        target_cols = [f'{temporal_config.location_target_field}_{i+1}' for i in range(len(map_object.G.nodes))]

        time_series = Requests_locations_time_series(train_data_df=train_data_df,
                                                     val_data_df=val_data_df,
                                                     test_data_df=test_data_df,
                                                     target_cols=target_cols,
                                                     temporal_config=temporal_config)
        
        slice_start_points_dict = {"train": train_slice_start_points_df["start_points"].tolist(),
                                "test": test_slice_start_points_df["start_points"].tolist(),
                                "val": val_slice_start_points_df["start_points"].tolist()}
        
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        train_prediction_df = self._generate_prediction_df(split="train",
                                                          time_series=time_series,
                                                          slice_start_points_dict=slice_start_points_dict,
                                                          data_df=train_data_df,
                                                          slice_start_points_df=train_slice_start_points_df,
                                                          device=device)
        
        val_prediction_df = self._generate_prediction_df(split="val",
                                                         time_series=time_series,
                                                         slice_start_points_dict=slice_start_points_dict,
                                                         data_df=val_data_df,
                                                         slice_start_points_df=val_slice_start_points_df,
                                                         device=device)
        
        test_prediction_df = self._generate_prediction_df(split="test",
                                                          time_series=time_series,
                                                          slice_start_points_dict=slice_start_points_dict,
                                                          data_df=test_data_df,
                                                          slice_start_points_df=test_slice_start_points_df,
                                                          device=device)

        prediction_df = pd.concat([train_prediction_df, val_prediction_df, test_prediction_df], axis=0, ignore_index=True)
        self._save_request_predictions(prediction_df=prediction_df)

        return prediction_df
    
    def _generate_prediction_df(self, split: str, time_series: Requests_locations_time_series,
                                slice_start_points_dict, data_df, slice_start_points_df, device):
        request_locations_dataset = Requests_locations_dataset(request_time_series=time_series,
                                                               slice_start_points_dict=slice_start_points_dict,
                                                               split=split,
                                                               sequence_length=self.num_intervals,
                                                               prediction_length=self.num_intervals)

        prediction_dict = self._generate_request_predictions(request_locations_dataset=request_locations_dataset,
                                                             test_data_df=data_df,
                                                             test_slice_start_points_df=slice_start_points_df,
                                                             device=device)

        prediction_df = pd.DataFrame(prediction_dict)
        return prediction_df
    
    def _load_model(self, device: torch.device):
        print("Loading request locations predictor ...")
        requests_locations_forecaster = Requests_locations_module.load_from_checkpoint(checkpoint_path=self.checkpoint_path)
        requests_locations_forecaster.to(device)
        print("Request locations predictor has been loaded!")
        
        return requests_locations_forecaster
    
    def _populate_prediction_dict(self, start_row, prediction_dict):
        prediction_dict["year"].append(start_row["year"])
        prediction_dict["month"].append(start_row["month_index"]+1)
        prediction_dict["day"].append(start_row["day_index"]+1)
        prediction_dict["hour"].append(start_row["hour"])
        local_interval_index = start_row["interval_index"] - ((start_row["hour"])*self.num_intervals)
        prediction_dict["interval_index"].append(local_interval_index)

        print(f"Generating predictions for {start_row["year"]}-{start_row["month_index"]+1}-{start_row["day_index"]+1}-{start_row["hour"]}-{local_interval_index}")
    
    def _generate_request_predictions(self, request_locations_dataset: Requests_locations_dataset,
                                      test_data_df: pd.DataFrame, test_slice_start_points_df: pd.DataFrame,
                                      device: torch.device):
        prediction_dict = {"year": [],
                           "month": [],
                           "day": [],
                           "hour": [],
                           "interval_index": [],
                           "prediction": []}
        
        requests_number_forecaster = self._load_model(device=device)

        print("Generating predictions for location distributionss ...")
        for i, start_point in enumerate(test_slice_start_points_df["start_points"]):
            start_row = test_data_df.iloc[start_point+request_locations_dataset.sequence_length]
            self._populate_prediction_dict(start_row=start_row, prediction_dict=prediction_dict)

            seq_x, seq_y, pos_w, pos_d, pos_day, pos_month = request_locations_dataset.__getitem__(i=i)

            seq_x = seq_x.unsqueeze(0)
            seq_y = seq_y.unsqueeze(0)
            pos_w = pos_w.unsqueeze(0)
            pos_d = pos_d.unsqueeze(0)
            pos_day = pos_day.unsqueeze(0)
            pos_month = pos_month.unsqueeze(0)

            true = seq_y.numpy()
            true = np.squeeze(true, axis=-1)
            shape = true.shape
            scaled_true = requests_number_forecaster.scaler.inverse_transform(true.reshape(shape[0] * shape[1], -1))

            pred = requests_number_forecaster.predict(seq_x=seq_x,
                                                      pos_w=pos_w,
                                                      pos_d=pos_d,
                                                      pos_day=pos_day,
                                                      pos_month=pos_month)
            
            pred = np.squeeze(pred, axis=0)

            rounded_pred = np.round(pred)
            sum_rounded = rounded_pred.sum(axis=1, keepdims=True)
            sum_rounded[sum_rounded == 0] = 1
            normalized_pred = rounded_pred / sum_rounded

            prediction_dict["prediction"].append(normalized_pred)
        
        print("Predicted location distributions have been generated!")

        return prediction_dict
    
    def _save_request_predictions(self, prediction_df: pd.DataFrame):
        print("Saving location predictions ...")
        prediction_df.to_pickle(os.path.join(self.data_folders.predicted_requests_folder_path, "request_locations.pkl"))
        prediction_df.to_csv(os.path.join(self.data_folders.predicted_requests_folder_path, "request_locations.csv"))
        print("Location predictions have been saved!")
    
    def _load_request_predictions(self):
        print("Loading location predictions ...")
        prediction_df = pd.read_pickle(os.path.join(self.data_folders.predicted_requests_folder_path, "request_locations.pkl"))
        print("Location predictions have been loaded!")
        return prediction_df
        

if __name__ == '__main__':
    """Performs execution delta of the process."""
    pStart = datetime.datetime.now()
    try:
        numbers_checkpoint_path = "data/STF_LOG_DIR/TimeMixer/TimeMixerepoch=45.ckpt"
        #numbers_checkpoint_path = "data/STF_LOG_DIR/iTransformer/iTransformerepoch=94.ckpt"
        # numbers_checkpoint_path = "data/STF_LOG_DIR/TimesNet/TimesNetepoch=121.ckpt"
        locations_checkpoint_path = "data/STF_LOG_DIR/STPGCN/STPGCNepoch=291.ckpt"
        num_intervals_number_predictions = 12
        num_intervals_locations_predictions = 12
        label_length = 0
        load_predictions_numbers = False
        load_predictions_locations = False

        data_folders = Data_folders()

        request_number_prediction_manager = Request_Number_Prediction_Manager(data_folders=data_folders,
                                                                              num_intervals=num_intervals_number_predictions,
                                                                              checkpoint_file_path=numbers_checkpoint_path,
                                                                              load_predictions=load_predictions_numbers,
                                                                              label_length=label_length)

        request_locations_prediction_manager = Request_Locations_Prediction_Manager(data_folders=data_folders,
                                                                                    num_intervals=num_intervals_locations_predictions,
                                                                                    checkpoint_file_path=locations_checkpoint_path,
                                                                                    load_predictions=load_predictions_locations)

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))  
