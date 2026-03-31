import numpy as np
import torch
import lightning as L

from bus_routing.Data_structures import Timeseries_model_config
from bus_routing.generative_models.time_series.architectures import TimesNet, TimeMixer, iTransformer, PatchTST
from bus_routing.generative_models.time_series.utils.metrics import metric

class Requests_number_module(L.LightningModule):

    def __init__(self, model_config: Timeseries_model_config, scaler):
        model_dict = {
            'TimesNet': TimesNet,
            'TimeMixer': TimeMixer,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST
        }
        super().__init__()
        self.model_config = model_config
        self.forecaster = model_dict[model_config.model_name].Model(self.model_config).float()
        self.scaler = scaler
        if self.model_config.loss == "MSE":
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.L1Loss()
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_config.learning_rate)
        return optimizer
    
    def _log_stats(self, section, outs):
        for key in outs.keys():
            stat = outs[key]
            if isinstance(stat, np.ndarray) or isinstance(stat, torch.Tensor):
                stat = stat.mean()
            self.log(f"{section}_{key}", stat, sync_dist=True, on_step=False, on_epoch=True)
    
    def _compute_stats(self, pred: torch.Tensor, true: torch.Tensor):
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()

        shape = pred.shape
        scaled_pred = self.scaler.inverse_transform(pred.reshape(shape[0] * shape[1], -1)).reshape(shape)
        scaled_true = self.scaler.inverse_transform(true.reshape(shape[0] * shape[1], -1)).reshape(shape)

        scaled_pred = scaled_pred[:, :, -1:]
        scaled_true = scaled_true[:, :, -1:]

        mae, mse, rmse = metric(scaled_pred, scaled_true)

        stats = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }

        return stats
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        dec_inp = torch.zeros_like(batch_y[:, -self.model_config.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.model_config.label_len, :], dec_inp], dim=1).float()

        outputs = self.forecaster(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1
        outputs = outputs[:, -self.model_config.pred_len:, :]
        batch_y = batch_y[:, -self.model_config.pred_len:, :]

        stats = self._compute_stats(pred=outputs, true=batch_y)

        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]

        loss = self.criterion(outputs, batch_y)

        stats["loss"] = loss

        return stats
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._log_stats(section="train",
                        outs=outputs)
        
        return outputs
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        dec_inp = torch.zeros_like(batch_y[:, -self.model_config.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.model_config.label_len, :], dec_inp], dim=1).float()

        outputs = self.forecaster(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1
        outputs = outputs[:, -self.model_config.pred_len:, :]
        batch_y = batch_y[:, -self.model_config.pred_len:, :]

        stats = self._compute_stats(pred=outputs, true=batch_y)

        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]
        
        loss = self.criterion(outputs, batch_y)

        stats["loss"] = loss

        return stats
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self._log_stats(section="val",
                        outs=outputs)
        
        return outputs
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        dec_inp = torch.zeros_like(batch_y[:, -self.model_config.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.model_config.label_len, :], dec_inp], dim=1).float()

        outputs = self.forecaster(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1
        outputs = outputs[:, -self.model_config.pred_len:, :]
        batch_y = batch_y[:, -self.model_config.pred_len:, :]

        stats = self._compute_stats(pred=outputs, true=batch_y)

        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]
        
        loss = self.criterion(outputs, batch_y)

        stats["loss"] = loss

        return stats
    
    def on_test_batch_end(self, outputs, batch, batch_idx):
        self._log_stats(section="test",
                        outs=outputs)
        
        return {"loss": outputs["loss"].mean()}
    
    def predict(self, x: torch.Tensor, x_mark: torch.Tensor, y_mark: torch.Tensor):

        x = x.to(self.device).float()
        x_mark = x_mark.to(self.device).float()
        y_mark = y_mark.to(self.device).float()

        dec_inp = torch.zeros_like(x[:, :, :], device=self.device).float()

        if self.model_config.label_len > 0:
            dec_inp = torch.cat([x[:, -self.model_config.label_len:, :], dec_inp], dim=1).float()

        with torch.no_grad():
            outputs = self.forecaster(x, x_mark, dec_inp, y_mark)

        f_dim = -1
        outputs = outputs[:, -self.model_config.pred_len:, :]

        pred = outputs.detach().cpu().numpy()

        shape = pred.shape
        scaled_pred = self.scaler.inverse_transform(pred.reshape(shape[0] * shape[1], -1)).reshape(shape)

        scaled_pred = scaled_pred[:, :, f_dim:]

        return scaled_pred



    

    
