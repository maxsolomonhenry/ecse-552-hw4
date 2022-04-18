import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class WeatherDataset(Dataset):

    def __init__(self, csv_path, n_past, to_predict = ["p (mbar)", "T (degC)", "rh (%)", "wv (m/s)"]):
        df = pd.read_csv(csv_path)
        df = self._parse_date(df)

        self._n_sequences = df.shape[0] - (n_past + 1)
        self._n_past = n_past

        self._y_indices = self._get_y_indices(df, to_predict)
        self._df = torch.tensor(df.values)

    def __getitem__(self, index):
        target_idx = index + self._n_past 

        p_in = index
        p_out = target_idx

        x = self._df[p_in:p_out, :]
        y = self._df[target_idx, self._y_indices]

        return x, y

    def __len__(self):
        return self._n_sequences

    def _get_y_indices(self, df, to_predict):
        indices = []

        for col_name in to_predict:
            indices.append(df.columns.get_loc(col_name))

        return torch.tensor(indices)

    def _parse_date(self, df):
        date_time = df["Date Time"]

        date = date_time.apply(lambda x: x.split(" ")[0])
        day = date.apply(lambda x: int(x.split(".")[0]))
        month = date.apply(lambda x: int(x.split(".")[1]))
        year = date.apply(lambda x: int(x.split(".")[2]))

        hour = date_time.apply(lambda x: x.split(" ")[-1])
        hour = hour.apply(lambda x: int(x.split(":")[0]))

        df.insert(0, "hour", hour)
        df.insert(0, "day", day)
        df.insert(0, "month", month)
        df.insert(0, "year", year)

        df = df.drop("Date Time", 1)

        return df

    def _remove_unwanted_features(self, df):
        return df

def get_dataloaders(fpath="data/weather_train.csv", n_past=8, batch_size=128, 
                    percent_train=0.8):

    dataset = WeatherDataset(fpath, n_past)

    n_train = int(percent_train * len(dataset))
    n_val = len(dataset) - n_train

    train_data, val_data = random_split(dataset, (n_train, n_val))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader