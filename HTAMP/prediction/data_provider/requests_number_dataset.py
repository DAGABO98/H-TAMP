import os
import torch
import datetime
import traceback
import pandas as pd

from torch.utils.data import Dataset
from sklearn.discriminant_analysis import StandardScaler

class RequestNumberDataManager:

    def __init__(self, data_folders: Data_folders, 
                 temporal_config: Temporal_data_config,
                 preprocess: bool = False,
                 save_data: bool = False,
                 ):
        
        self.data_folders = data_folders
        self.request_handler = Request_handler(data_folders=data_folders,
                                               process_requests=preprocess)
        self.temporal_config = temporal_config

        if preprocess:
            print("Preprocessing Weather Data ...")
            weather_features_df = _preprocess_weather_data(temporal_config=self.temporal_config,
                                                           data_folders=self.data_folders)
            print("Weather data has been processed!")
            
            print("Preprocessing Number of Requests Data ...")
            number_requests_data = self._preprocess_number_requests_data(weather_features_df=weather_features_df)
            self._unpack_preprocess_data(number_requests_data=number_requests_data)
            print("Number of requests data has been processed!")
            if save_data:
                self._save_dataframes()
        else:
            self._load_dataframes()
    
    def _unpack_preprocess_data(self, number_requests_data):
        train_number_requests_data = number_requests_data[0]
        self.train_number_of_requests_df, self.train_number_requests_slice_indices_df = train_number_requests_data
        val_number_requests_data = number_requests_data[1]
        self.val_number_of_requests_df, self.val_number_requests_slice_indices_df = val_number_requests_data
        test_number_requests_data = number_requests_data[2]
        self.test_number_of_requests_df, self.test_number_requests_slice_indices_df = test_number_requests_data
    
    def _save_dataframes(self):
        print("Saving traing and testing data ...")
        self.train_number_of_requests_df.to_csv(os.path.join(self.temporal_config.requests_number_folder, "train_data.csv"), index=False)
        self.val_number_of_requests_df.to_csv(os.path.join(self.temporal_config.requests_number_folder, "val_data.csv"), index=False)
        self.test_number_of_requests_df.to_csv(os.path.join(self.temporal_config.requests_number_folder, "test_data.csv"), index=False)
        self.train_number_requests_slice_indices_df.to_csv(os.path.join(self.temporal_config.requests_number_folder, "train_slices_indices.csv"), index=False)
        self.val_number_requests_slice_indices_df.to_csv(os.path.join(self.temporal_config.requests_number_folder, "val_slices_indices.csv"), index=False)
        self.test_number_requests_slice_indices_df.to_csv(os.path.join(self.temporal_config.requests_number_folder, "test_slices_indices.csv"), index=False)
        print("Training and testing data saved!")

    def _load_dataframes(self):
        print("Loading traing and testing data ...")
        self.train_number_of_requests_df = pd.read_csv(os.path.join(self.temporal_config.requests_number_folder, "train_data.csv"))
        self.val_number_of_requests_df = pd.read_csv(os.path.join(self.temporal_config.requests_number_folder, "val_data.csv"))
        self.test_number_of_requests_df = pd.read_csv(os.path.join(self.temporal_config.requests_number_folder, "test_data.csv"))
        self.train_number_requests_slice_indices_df = pd.read_csv(os.path.join(self.temporal_config.requests_number_folder, "train_slices_indices.csv"))
        self.val_number_requests_slice_indices_df = pd.read_csv(os.path.join(self.temporal_config.requests_number_folder, "val_slices_indices.csv"))
        self.test_number_requests_slice_indices_df = pd.read_csv(os.path.join(self.temporal_config.requests_number_folder, "test_slices_indices.csv"))
        print("Training and testing data loaded!")

    def _initialize_number_requests_dictionary(self):
        number_of_requests_values = {}

        for weather_field in self.temporal_config.weather_fields:
            number_of_requests_values[weather_field] = []
        
        for time_field in self.temporal_config.time_fields:
            number_of_requests_values[time_field] = []
        
        for num_requests_target_field in self.temporal_config.num_requests_target_fields:
            number_of_requests_values[num_requests_target_field] = []
        
        return number_of_requests_values
    
    def _populate_context_varirables_for_num_requests(self, year: int, month: int, day: int, date_weekday_index: int, hour: int, 
                                                      minute_interval_index: int,current_weather_series: pd.Series, 
                                                      number_of_requests_values: dict[str, list[int]]):
        for weather_field in self.temporal_config.weather_fields:
            number_of_requests_values[weather_field].append(current_weather_series[weather_field])

        number_of_requests_values['year'].append(year)
        number_of_requests_values['month'].append(month)
        number_of_requests_values['day'].append(day)
        number_of_requests_values['weekday'].append(date_weekday_index)
        number_of_requests_values['hour'].append(hour)
        number_of_requests_values['interval_index'].append(minute_interval_index)
    
    def _determine_initial_date_time_elements(self, date_operational_range: Date_operational_range, 
                                              year_range, month_range, day_range):
        if self.temporal_config.start_hour - 1 < 0:
            prev_hour = 23
            if day_range[0] - 1 < 1:
                if month_range[0] - 1 < 1:
                    prev_month = 12
                    prev_year = year_range[0] - 1
                else:
                    prev_month = month_range[0] - 1
                    prev_year = year_range[0]
                
                prev_day = date_operational_range.month_lengths[prev_year][prev_month-1]
            else:
                prev_year = year_range[0]
                prev_month = month_range[0]
                prev_day = day_range[0]
        else:
            prev_hour = self.temporal_config.start_hour - 1
            prev_year = year_range[0]
            prev_month = month_range[0]
            prev_day = day_range[0]
        
        return prev_year, prev_month, prev_day, prev_hour
    
    def _determine_final_date_time_elements(self, date_operational_range: Date_operational_range, 
                                            year_range: list[int], month_range: list[int], 
                                            day_range: list[int], hour_range: list[int], last_index: int):
        if hour_range[last_index] + 1 > 23:
            following_hour = 0
            if day_range[last_index] + 1 > date_operational_range.month_lengths[year_range[last_index]][month_range[last_index]-1]:
                following_day = 1
                if date_operational_range.month + 1 > 12:
                    following_month = 1
                    following_year = year_range[last_index] + 1
                else:
                    following_month = month_range[last_index] + 1
                    following_year = year_range[last_index]
            else:
                following_day = day_range[last_index] + 1
                following_month = month_range[last_index]
                following_year = year_range[last_index]
        else:
            following_hour = hour_range[last_index] + 1
            following_day = day_range[last_index]
            following_month = month_range[last_index]
            following_year = year_range[last_index]
        
        return following_year, following_month, following_day, following_hour
    
    def _initialize_number_requests_slice(self, prev_year, prev_month, prev_day, prev_hour, minute_intervals,
                                          weather_features_df, number_of_requests_values):
        new_date_string = str(prev_year)+"-"+str(prev_month)+"-"+str(prev_day)
        new_date_object = pd.to_datetime(new_date_string).date()
        date_weekday_index = new_date_object.weekday()

        weather_retrieval_mask = (weather_features_df["date"].dt.date == new_date_object) \
                                    & (weather_features_df["hour"] == prev_hour)
        
        current_weather_series = weather_features_df[weather_retrieval_mask].iloc[0]

        for minute_interval_index in range(len(minute_intervals)):

            self._populate_context_varirables_for_num_requests(year=prev_year,
                                                               month=prev_month,
                                                               day=prev_day,
                                                               date_weekday_index=date_weekday_index,
                                                               hour=prev_hour,
                                                               minute_interval_index=minute_interval_index,
                                                               current_weather_series=current_weather_series,
                                                               number_of_requests_values=number_of_requests_values)

            number_of_requests_values["number_requests"].append(0)
    
    def _finalize_number_requests_slice(self, next_year, next_month, next_day, next_hour, minute_intervals,
                                        weather_features_df, number_of_requests_values):
        new_date_string = str(next_year)+"-"+str(next_month)+"-"+str(next_day)
        new_date_object = pd.to_datetime(new_date_string).date()
        date_weekday_index = new_date_object.weekday()

        weather_retrieval_mask = (weather_features_df["date"].dt.date == new_date_object) \
                                    & (weather_features_df["hour"] == next_hour)
        
        current_weather_series = weather_features_df[weather_retrieval_mask].iloc[0]

        for minute_interval_index in range(len(minute_intervals)):

            self._populate_context_varirables_for_num_requests(year=next_year,
                                                               month=next_month,
                                                               day=next_day,
                                                               date_weekday_index=date_weekday_index,
                                                               hour=next_hour,
                                                               minute_interval_index=minute_interval_index,
                                                               current_weather_series=current_weather_series,
                                                               number_of_requests_values=number_of_requests_values)

            number_of_requests_values["number_requests"].append(0)

    def _populate_number_requests_slices(self, weather_features_df, online_requests_df, date_element, minute_intervals, 
                                         number_of_requests_values, number_requests_slice_indices: list[int], 
                                         num_requests_slice_index: int):

        date_operational_range = Date_operational_range(year=date_element.year,
                                                        month=date_element.month,
                                                        day=date_element.day,
                                                        start_hour=self.temporal_config.start_hour,
                                                        end_hour=self.temporal_config.end_hour)
            
        hour_range, day_range, month_range, year_range = date_operational_range.get_operating_ranges()
        
        prev_year, prev_month, prev_day, prev_hour = self._determine_initial_date_time_elements(date_operational_range=date_operational_range,
                                                                                                year_range=year_range,
                                                                                                month_range=month_range,
                                                                                                day_range=day_range)
        
        self._initialize_number_requests_slice(prev_year=prev_year,
                                               prev_month=prev_month,
                                               prev_day=prev_day,
                                               prev_hour=prev_hour,
                                               minute_intervals=minute_intervals,
                                               weather_features_df=weather_features_df,
                                               number_of_requests_values=number_of_requests_values)

        for i, hour_of_interest in enumerate(hour_range):

            new_date_string = str(year_range[i])+"-"+str(month_range[i])+"-"+str(day_range[i])
            new_date_object = pd.to_datetime(new_date_string).date()
            date_weekday_index = new_date_object.weekday()

            weather_retrieval_mask = (weather_features_df["date"].dt.date == new_date_object) \
                                        & (weather_features_df["hour"] == hour_of_interest)
            
            current_weather_series = weather_features_df[weather_retrieval_mask].iloc[0]
            requests_for_current_hour = 0

            for minute_interval_index, minute_interval in enumerate(minute_intervals):
                number_requests_slice_indices.append(num_requests_slice_index)

                num_requests_slice_index += 1

                self._populate_context_varirables_for_num_requests(year=year_range[i],
                                                                   month=month_range[i],
                                                                   day=day_range[i],
                                                                   date_weekday_index=date_weekday_index,
                                                                   hour=hour_of_interest,
                                                                   minute_interval_index=minute_interval_index,
                                                                   current_weather_series=current_weather_series,
                                                                   number_of_requests_values=number_of_requests_values)

                retrieval_mask = (online_requests_df["Requested Pickup Time"].dt.date == new_date_object) \
                    & (online_requests_df["Requested Pickup Time"].dt.hour == hour_of_interest) \
                    & (online_requests_df["Requested Pickup Time"].dt.minute >= minute_interval[0]) \
                    & (online_requests_df["Requested Pickup Time"].dt.minute <= minute_interval[1])
                
                retrieved_requests = online_requests_df[retrieval_mask]
                number_of_requests_in_interval = len(retrieved_requests.index)

                number_of_requests_values["number_requests"].append(number_of_requests_in_interval)
                requests_for_current_hour += number_of_requests_in_interval
        
        next_year, next_month, next_day, next_hour = self._determine_final_date_time_elements(date_operational_range=date_operational_range,
                                                                                              year_range=year_range,
                                                                                              month_range=month_range,
                                                                                              day_range=day_range,
                                                                                              hour_range=hour_range,
                                                                                              last_index=len(hour_range)-1)
        
        self._finalize_number_requests_slice(next_year=next_year,
                                             next_month=next_month,
                                             next_day=next_day,
                                             next_hour=next_hour,
                                             minute_intervals=minute_intervals,
                                             weather_features_df=weather_features_df,
                                             number_of_requests_values=number_of_requests_values)
            
        num_requests_slice_index += 2 * len(minute_intervals)
        return num_requests_slice_index
    
    def _process_number_requests_data(self, weather_features_df: pd.DataFrame, online_requests_df: pd.DataFrame, 
                                      unique_date_elements, minute_intervals, split: str = "train"):
        number_of_requests_values = self._initialize_number_requests_dictionary()
        number_requests_slice_indices = []
        number_requests_slice_index = 0

        print(f"Generating data for {split} set")

        for date_element in unique_date_elements:
            print("Processing requests for " + str(date_element))
            number_requests_slice_index = self._populate_number_requests_slices(weather_features_df=weather_features_df,
                                                                                online_requests_df=online_requests_df,
                                                                                date_element=date_element,
                                                                                minute_intervals=minute_intervals,
                                                                                number_of_requests_values=number_of_requests_values,
                                                                                number_requests_slice_indices=number_requests_slice_indices,
                                                                                num_requests_slice_index=number_requests_slice_index)
            
        return number_of_requests_values, number_requests_slice_indices
    
    def _split_dates(self):
        unique_date_elements = list(self.request_handler.online_requests_df["Request Creation Date"].unique())
        unique_date_elements_set = set(unique_date_elements)

        testing_dates = _generate_testing_dates(unique_date_elements=unique_date_elements)
        testing_dates_set = set(testing_dates)

        remaining_date_elements_set = unique_date_elements_set - testing_dates_set

        validation_dates = _generate_validation_dates(unique_date_elements=list(remaining_date_elements_set))
        validation_dates_set = set(validation_dates)

        training_dates_set = remaining_date_elements_set - validation_dates_set
        training_dates = list(training_dates_set)

        return testing_dates, validation_dates, training_dates
    
    def _generate_dataframes(self, weather_features_df, unique_date_elements, minute_intervals, split: str):
        requests_data = self._process_number_requests_data(weather_features_df=weather_features_df,
                                                           online_requests_df=self.request_handler.online_requests_df,
                                                           unique_date_elements=unique_date_elements,
                                                           minute_intervals=minute_intervals,
                                                           split=split)
        
        number_of_requests_values = requests_data[0]
        number_requests_slice_indices = requests_data[1]

        num_reqs_flag, num_reqs_lengths = _check_lists_lengths(dictionary=number_of_requests_values)
        assert num_reqs_flag, f"All entries should have a field in every column! lengths = {num_reqs_lengths}"

        print(f"{split} data has been generated!")

        number_of_requests_df = pd.DataFrame(number_of_requests_values)
        number_requests_slice_indices_df = pd.DataFrame(number_requests_slice_indices, columns=["start_points"])
        number_requests_data = (number_of_requests_df, number_requests_slice_indices_df)

        return number_requests_data
        
    def _preprocess_number_requests_data(self, weather_features_df: pd.DataFrame):
        test_dates, val_dates, train_dates = self._split_dates()

        minute_intervals = self.request_handler.generate_minute_intervals(interval_length=self.temporal_config.interval_length)

        train_number_requests_data = self._generate_dataframes(weather_features_df=weather_features_df,
                                                               unique_date_elements=train_dates,
                                                               minute_intervals=minute_intervals,
                                                               split="train")
        
        val_number_requests_data = self._generate_dataframes(weather_features_df=weather_features_df,
                                                             unique_date_elements=val_dates,
                                                             minute_intervals=minute_intervals,
                                                             split="val")
        
        test_number_requests_data = self._generate_dataframes(weather_features_df=weather_features_df,
                                                              unique_date_elements=test_dates,
                                                              minute_intervals=minute_intervals,
                                                              split="test")
        
        number_requests_data = (train_number_requests_data, val_number_requests_data, test_number_requests_data)
        
        return number_requests_data

    def get_requests_numbers_training_data(self):
        return self.train_number_of_requests_df, self.train_number_requests_slice_indices_df
    
    def get_requests_numbers_validation_data(self):
        return self.val_number_of_requests_df, self.val_number_requests_slice_indices_df
    
    def get_requests_numbers_testing_data(self):
        return self.test_number_of_requests_df, self.test_number_requests_slice_indices_df
        

class Requests_numbers_time_series:
    def __init__(self, train_data_df: pd.DataFrame, val_data_df: pd.DataFrame, test_data_df: pd.DataFrame, 
                 temporal_config: Temporal_data_config):
        self.train_data_df = train_data_df
        self.val_data_df = val_data_df
        self.test_data_df = test_data_df
        self.time_cols = temporal_config.time_fields
        self.weather_cols = temporal_config.weather_fields
        self.target_cols = temporal_config.num_requests_target_fields
        self.temporal_config = temporal_config
        request_scalers = Requests_temp_scalers(temporal_config=temporal_config)
        self.min_values_dict = request_scalers.get_min_values_dict()
        self.max_values_dict = request_scalers.get_max_values_dict()
        self.target_scaler = StandardScaler()
        scaled_data = self.apply_target_scaling_df(train_df=self.train_data_df,
                                                   val_df=self.val_data_df,
                                                   test_df=self.test_data_df)
        scaled_train_data, scaled_val_data, scaled_test_data = scaled_data
        self.scaled_train_data_df = self.apply_temporal_scaling_df(df=scaled_train_data)
        self.scaled_val_data_df = self.apply_temporal_scaling_df(df=scaled_val_data)
        self.scaled_test_data_df = self.apply_temporal_scaling_df(df=scaled_test_data)
    
    def apply_target_scaling_df(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        train_scaled = train_df.copy(deep=True)
        val_scaled = val_df.copy(deep=True)
        test_scaled = test_df.copy(deep=True)

        combined_cols = self.weather_cols + self.target_cols

        self.target_scaler.fit(train_scaled[combined_cols])
        train_scaled[combined_cols] = self.target_scaler.transform(train_scaled[combined_cols])
        val_scaled[combined_cols] = self.target_scaler.transform(val_scaled[combined_cols])
        test_scaled[combined_cols] = self.target_scaler.transform(test_scaled[combined_cols])

        return train_scaled, val_scaled, test_scaled
    
    def apply_temporal_scaling_df(self, df: pd.DataFrame):
        # temporal features are in the range [-0.5, 0.5]
        scaled = df.copy(deep=True)
        cols = self.time_cols

        min_elements = []
        max_elements = []
        for column in cols:
            min_elements.append(self.min_values_dict[column])
            max_elements.append(self.max_values_dict[column])

        scaled[cols] = ((df[cols] - pd.array(min_elements)) / (pd.array(max_elements) - pd.array(min_elements))) - 0.5
        return scaled

    def get_slice(self, split, start, stop):
        assert split in ["train", "val", "test"]
        if split == "train":
            return self.train_data.iloc[start:stop]
        elif split == "val":
            return self.val_data.iloc[start:stop]
        else:
            return self.test_data.iloc[start:stop]

    @property
    def train_data(self):
        return self.scaled_train_data_df

    @property
    def val_data(self):
        return self.scaled_val_data_df

    @property
    def test_data(self):
        return self.scaled_test_data_df

    def length(self, split):
        return {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }[split]


class Requests_numbers_dataset(Dataset):
    def __init__(
        self,
        request_time_series: Requests_numbers_time_series,
        slice_start_points_dict,
        split: str = "train",
        sequence_length: int = 60,
        label_length: int  = 10,
        prediction_length: int = 60,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.series = request_time_series
        self.sequence_length = sequence_length
        self.label_length = label_length
        self.prediction_length = prediction_length
        self._slice_start_points = slice_start_points_dict[split]

    def __len__(self):
        return len(self._slice_start_points)

    def _torch(self, *dfs):
        return tuple(torch.from_numpy(x.values).float() for x in dfs)

    def __getitem__(self, i):
        start = self._slice_start_points[i]
        series_slice = self.series.get_slice(self.split, 
                                             start=start, 
                                             stop=start + (self.sequence_length + self.prediction_length))
        
        x_slice, y_slice = (
            series_slice.iloc[: self.sequence_length],
            series_slice.iloc[self.sequence_length - self.label_length :],
        )

        seq_x = x_slice[self.series.weather_cols + self.series.target_cols]
        seq_x_mark = x_slice[self.series.time_cols]

        seq_y = y_slice[self.series.weather_cols + self.series.target_cols]
        seq_y_mark = y_slice[self.series.time_cols]

        return self._torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
    
    def inverse_transform(self, data):
        return self.series.target_scaler.inverse_transform(data)
    
if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        preprocess = True
        pred_len = 12
        interval_length = 60 // pred_len
        data_folders = Data_folders()
        temporal_config = Temporal_data_config(interval_length=interval_length)

        if preprocess:
            request_data_manager = Request_Number_Data_Manager(data_folders=data_folders,
                                                        temporal_config=temporal_config,
                                                        preprocess=True,
                                                        save_data=True)
        else:
            request_data_manager = Request_Number_Data_Manager(data_folders=data_folders,
                                                    temporal_config=temporal_config,
                                                    preprocess=False,
                                                    save_data=False)
        
        train_data_df, train_slice_start_points_df = request_data_manager.get_requests_numbers_training_data()
        val_data_df, val_slice_start_points_df = request_data_manager.get_requests_numbers_validation_data()
        test_data_df, test_slice_start_points_df = request_data_manager.get_requests_numbers_testing_data()

        dset = Requests_numbers_time_series(train_data_df=train_data_df,
                                            val_data_df=val_data_df,
                                            test_data_df=test_data_df,
                                            temporal_config=temporal_config)
        
        slice_start_points_dict = {"train": train_slice_start_points_df["start_points"].tolist(),
                                "test": test_slice_start_points_df["start_points"].tolist(),
                                "val": val_slice_start_points_df["start_points"].tolist()}
        DATA_MODULE = Requests_data_module(
        datasetCls=Requests_numbers_dataset,
        dataset_kwargs={
            "request_time_series": dset,
            "slice_start_points_dict": slice_start_points_dict,
            "sequence_length": pred_len,
            "label_length": pred_len//4,
            "prediction_length": pred_len
        },
        batch_size=32,
        workers=4,
        collate_fun=None
        )

        test_data_loader = DATA_MODULE.test_dataloader()

        for i, batch in enumerate(test_data_loader):
            seq_x, seq_y, seq_x_mark, seq_y_mark = batch
            print(i)
            print(seq_x.size())
            print(seq_y.size())
            print(seq_x_mark.size())
            print(seq_y_mark.size())

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart)) 