import copy
from pathlib import Path
import warnings
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tensorboard as tb
import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, RMSE, QuantileLoss, MultivariateNormalDistributionLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data import TorchNormalizer

from dataloader import TimeSeriesLoader

# Source for much of this code from the `pytorch-forecasting` TFT tutorial, found here:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html


def create_datasets(
        dataset,
        time_col='ID',
        target_y='yl_t+60(val)',
        target_y_flag='yl_t+60(flag)',
        max_prediction_length=24,
        max_encoder_length=48,
        train_month_cutoff=10,
        val_month_cutoff=12,
):
    dataset_df = dataset[dataset[target_y_flag] == 1].reset_index(drop=True)
    dataset_df[time_col] = dataset_df.index

    train_cutoff_idx = dataset_df[dataset_df['month_day'] >= train_month_cutoff][time_col].values[0]
    val_cutoff_idx = dataset_df[dataset_df['month_day'] >= val_month_cutoff][time_col].values[0]
    print('training cutoff idx:', train_cutoff_idx)
    print('validation cutoff idx:', val_cutoff_idx)

    training_cutoff = train_cutoff_idx - max_prediction_length

    training = TimeSeriesDataSet(
        dataset_df[lambda x: x[time_col] <= training_cutoff],
        time_idx=time_col,
        target=target_y,
        group_ids=['constant'],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=['holiday', 'weekday'],
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=['month_day'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            'DHI',
            'DNI',
            'GHI',
            'Dew Point',
            'Solar Zenith Angle',
            'Wind Speed',
            'Relative Humidity',
            'Temperature',
            target_y,
        ],
        # target_normalizer=TorchNormalizer(
        #     method='identity', center=True, transformation=None, method_kwargs={}
        # ), # https://github.com/jdb78/pytorch-forecasting/issues/1220
        # add_relative_time_idx=True,
        add_target_scales=True,
        # add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training,
        dataset_df,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=train_cutoff_idx + 1,
    )

    testing = TimeSeriesDataSet.from_dataset(
        training,
        dataset_df,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=val_cutoff_idx - 1,
    )

    return training, validation, testing


def create_dataloaders(training, validation, testing, batch_size=128, device='cpu'):
    use_pin_memory = device != 'cpu'

    # create dataloaders for model
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, pin_memory=use_pin_memory, batch_sampler='synchronized')
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0, pin_memory=use_pin_memory, batch_sampler='synchronized')
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0, pin_memory=use_pin_memory, batch_sampler='synchronized')

    return train_dataloader, val_dataloader, test_dataloader


def get_val_test_actuals(val_dataloader, test_dataloader, device='cpu'):
    val_actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).to(device)
    test_actuals = torch.cat([y[0] for x, y in iter(test_dataloader)]).to(device)

    return val_actuals, test_actuals


def calculate_error(predictions, actuals, dataset_str, year, device='cpu'):
    print()

    rmse = RMSE().to(device)
    rmse_val = rmse(predictions, actuals)
    rmse_string = f'{year} {dataset_str} RMSE: {rmse_val}\n'
    print(rmse_string)

    mae = MAE().to(device)
    mae_val = mae(predictions, actuals)
    mae_string = f'{year} {dataset_str} MAE: {mae_val}\n'
    print(mae_string)

    return rmse_string, mae_string


def plot_learning_curves(train_lc_csv_path, val_lc_csv_path, year):
    train_lc = pd.read_csv(train_lc_csv_path).drop('Wall time', axis=1)
    val_lc = pd.read_csv(val_lc_csv_path).drop('Wall time', axis=1)

    # merge dfs into one for easier export
    lc_df = train_lc.merge(val_lc, on='Step', suffixes=('_train', '_val'))
    lc_path = f'./learning_curves/learning_curves_{year}'
    lc_df.to_csv(f'{lc_path}.csv')

    plt.plot(lc_df['Step'].to_numpy(), lc_df['Value_train'].to_numpy(), label='Training')
    plt.plot(lc_df['Step'].to_numpy(), lc_df['Value_val'].to_numpy(), label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, )
    plt.title(f'Train/Val Learning Curves: {year}')
    plt.legend()
    plt.savefig(f'{lc_path}.png')
    plt.show()
    plt.clf()


def main():
    if torch.cuda.is_available():
        device = 'cuda'
        pl_accelerator = 'gpu'
    else:
        device = 'cpu'
        pl_accelerator = 'cpu'

    # set to True if raw data has already been preprocessed
    data_preprocessed = False

    if not data_preprocessed:
        loader = TimeSeriesLoader(
            task='forecasting',
            root='../PSML',  # suppose the raw dataset is downloaded and unzipped under Open-source-power-dataset
        )
        train_loader, test_loader = loader.load(batch_size=32, shuffle=True)

    # orig_csv_path = '../PSML/Minute-level Load and Renewable/CAISO_zone_1_.csv'
    # data_original = pd.read_csv(orig_csv_path)

    root = '../PSML'
    data_folder = os.path.join(root, 'processed_dataset', 'forecasting')
    location = 'CAISO_zone_1'
    years = ['2018', '2019', '2020']

    data = pd.DataFrame()
    data_dfs = []

    for year in years:
        data_append = pd.read_csv(os.path.join(data_folder, f'{location}_{year}.csv'))

        # turn holiday into categorical
        data_append['holiday'] = pd.Categorical(data_append['holiday'].astype(str))
        data_append['weekday'] = pd.Categorical(data_append['weekday'].astype(str))
        data_append['constant'] = 'filler'

        data_dfs.append(data_append)
        data = pd.concat((data, data_append), ignore_index=True)

    data_2018 = data_dfs[0]
    data_2019 = data_dfs[1]
    data_2020 = data_dfs[2]

    time_col = 'ID'
    data[time_col] = data.index

    # set learning rate
    optimal_lr = 0.0275

    pl.seed_everything(42)

    loss_strings = []

    for i, year in enumerate(years):

        training, validation, testing = create_datasets(data_dfs[i])
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(training, validation, testing, device=device)
        val_actuals, test_actuals = get_val_test_actuals(val_dataloader, test_dataloader, device=device)

        # calculate baseline MAE and RMSE
        baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator=pl_accelerator))
        rmse_baseline, mae_baseline = calculate_error(baseline_predictions, val_actuals, 'Baseline', year, device=device)
        loss_strings.extend([rmse_baseline, mae_baseline])

        # configure network and trainer

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        trainer = pl.Trainer(
            # max_epochs=25,
            accelerator=pl_accelerator,
            enable_model_summary=True,
            gradient_clip_val=0.077,
            limit_train_batches=50,  # comment in for training, running validation every x batches
            # fast_dev_run=True,  # comment in to check that network or dataset has no serious bugs
            limit_val_batches=50,
            callbacks=[lr_logger, early_stop_callback],
            enable_checkpointing=True,
            logger=logger,
        )

        # filled in with HPs from HP tuning
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=optimal_lr,
            hidden_size=11,
            attention_head_size=1,
            dropout=0.2,
            hidden_continuous_size=9,  # set to <= hidden_size
            loss=QuantileLoss(),
            log_interval=10,
            optimizer="Ranger",
            reduce_on_plateau_patience=4,  # reduce learning rate if no improvement in validation loss after x epochs
        )
        print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        # minor precision sacrifice for major performance gains on CUDA-enabled GPU
        if device != 'cpu':
            torch.set_float32_matmul_precision('high')

        # fit network
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # calculate RMSE and MAE on validation set
        val_predictions = tft.predict(val_dataloader, trainer_kwargs=dict(accelerator="gpu"))
        rmse_val, mae_val = calculate_error(val_predictions, val_actuals, 'Validation', year, device=device)
        loss_strings.extend([rmse_val, mae_val])

        # manual step - haven't figured out how to automate this:
        # open TensorBoard, download correct learning curves, and rename to the expected format
        input("Manual step: download correct learning curves from TB and rename (Y when done): ")

        # plot learning curves
        train_lc_csv_path = f'./pl_logs/train_loss_{year}.csv'
        val_lc_csv_path = f'./pl_logs/val_loss_{year}.csv'
        plot_learning_curves(train_lc_csv_path, val_lc_csv_path, year)

        # calculate test set error
        test_predictions = tft.predict(test_dataloader, trainer_kwargs=dict(accelerator="gpu"))
        rmse_test, mae_test = calculate_error(test_predictions, test_actuals, 'Test', year, device=device)
        loss_strings.extend([rmse_test, mae_test])

        input("Continue (y): ")

    # write loss strings to new file
    with open('results.txt', 'w') as f:
        f.writelines(loss_strings)


if __name__ == '__main__':
    main()
