import os
import wandb
import argparse
import datetime
import traceback

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from bus_routing.Map_graph import Map_graph
from bus_routing.Data_structures import Data_folders, Temporal_data_config, Timeseries_model_config

from bus_routing.generative_models.Requests_number_module import Requests_number_module
from bus_routing.generative_models.data_provider.Requests_number_dataset import Requests_numbers_dataset, Requests_data_module
from bus_routing.generative_models.data_provider.Requests_number_dataset import Request_Number_Data_Manager, Requests_numbers_time_series

class Requests_number_predictor:
    
    def create_data_module_and_scaler(self, model_config: Timeseries_model_config):
        data_folders = Data_folders()
        interval_length = 60 // model_config.pred_len
        temporal_config = Temporal_data_config(interval_length=interval_length)
        if model_config.preprocess_data:
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
        data_module = Requests_data_module(
        datasetCls=Requests_numbers_dataset,
        dataset_kwargs={
            "request_time_series": dset,
            "slice_start_points_dict": slice_start_points_dict,
            "sequence_length": model_config.seq_len,
            "label_length": model_config.label_len,
            "prediction_length": model_config.pred_len
        },
        batch_size=model_config.batch_size,
        workers=model_config.num_workers,
        collate_fun=None
        )

        return data_module, dset.target_scaler
    
    def create_model(self, model_config, scaler):
        forecaster = Requests_number_module(model_config=model_config,
                                            scaler=scaler)
        
        return forecaster

    def create_callbacks(self, model_config, save_dir):
        filename = f"{model_config.run_name}"
        model_ckpt_dir = os.path.join(save_dir, filename)
        saving = ModelCheckpoint(dirpath=model_ckpt_dir,
                                 monitor="val_loss",
                                 mode="min",
                                 filename=f"{model_config.run_name}" + "{epoch:02d}",
                                 save_top_k=1,
                                 auto_insert_metric_name=True)
        callbacks = [saving]

        callbacks.append(EarlyStopping(monitor="val_loss",
                                        patience=model_config.patience))

        callbacks.append(LearningRateMonitor())

        return callbacks
    
    def compile_and_train(self, args):
        model_config = Timeseries_model_config(args=args)
        log_dir = os.getenv("STF_LOG_DIR")
        if log_dir is None:
            log_dir = "./data/STF_LOG_DIR"
            print(
                "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
            )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if model_config.wandb:
            experiment = wandb.init(project="timeseries_predictor",
                                    entity="react_lab",
                                    config=args,
                                    dir=log_dir,
                                    reinit=True)
            config = wandb.config
            wandb.run.name = args.run_name
            wandb.run.save()
            logger = WandbLogger(experiment=experiment,
                                save_dir=log_dir)
            logger.log_hyperparams(config)
        else:
            logger = None

        data_module, scaler = self.create_data_module_and_scaler(model_config=model_config)

        forecaster = self.create_model(model_config=model_config, scaler=scaler)

        callbacks = self.create_callbacks(model_config=model_config, save_dir=log_dir)

        if model_config.model_name == "TimeMixer":
            strategy='ddp_find_unused_parameters_true'
        else:
            strategy='auto'

        trainer = Trainer(devices=[1,2],
                          accelerator="cuda",
                          strategy=strategy,
                          callbacks=callbacks,
                          logger=logger,
                          max_epochs=model_config.max_epochs)

        # Train
        trainer.fit(forecaster, datamodule=data_module)

        # Test
        trainer.test(datamodule=data_module, ckpt_path="best")

        if model_config.wandb:
            experiment.finish()


if __name__ == '__main__':
    """Performs execution delta of the process."""
    # Unit tests
    pStart = datetime.datetime.now()
    try:
        parser = argparse.ArgumentParser(prog="RequestsPredictor",
                                         description="Script for training a deep learning model for predicting requests")
        parser.add_argument('--model_name', type=str, default='TimesNet',
                            help='model name, options: [TimesNet, TimeMixer, iTransformer, PatchTST]')
        parser.add_argument('--run_name', type=str, default='TimesNet_test',
                            help='Name for the current run')
        parser.add_argument("--preprocess_data", action='store_true', default=False, 
                            help='Flag for preprocessing data before loading it.')
        parser.add_argument("--wandb", action='store_true', default=False, 
                            help='Flag for using wandb for logging training progress')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

        # model define
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=4, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=4, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')

        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--channel_independence', type=int, default=0,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')

        # optimization
        parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
        parser.add_argument('--max_epochs', type=int, default=300, help='max number of train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=40, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')

        args = parser.parse_args()
        request_pred = Requests_number_predictor()
        request_pred.compile_and_train(args=args)

    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop-pStart))  
