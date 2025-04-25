import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input

def process_data(input_file):
    data = pd.read_excel(input_file)
    data = data.loc[5:, :]
    data = data.drop('Unnamed: 2', axis=1)
    data.columns = ['time', 'value']
    data['time'] = pd.to_datetime(data['time'])
    data['value'] = data['value'].str.replace(',', '.')
    data['value'] = data['value'].astype(float)
    data = data.set_index('time')
    data = data.resample('S').mean()
    data = data.fillna(method='ffill')
    data['seconds'] =  data.index.hour * 3600 + data.index.minute * 60 + data.index.second
    return data



def prediction_plot(df, start_second, hours, train_time_seconds, model, scaler):
    selected_data = df.query('seconds >= @start_second and seconds <= @start_second + @train_time_seconds - 1')

    n_features = 1

    real_data_after_train = df.query('seconds > @start_second + @train_time_seconds - 1')

    scaled_selected = scaler.transform(selected_data[['value']])

    last_hour_data = scaled_selected

    X_new = last_hour_data.reshape((1, 2700, n_features))

    predictions = model.predict(X_new)

    predicted_values = scaler.inverse_transform(predictions)

    train_start = start_second
    train_end = train_start + train_time_seconds - 1

    train_data = selected_data

    prediction_start = train_end
    prediction_end = prediction_start + hours * 3600
    predictions_data = predicted_values.flatten()

    predicted_seconds = np.arange(train_time_seconds, hours * 3600 + train_time_seconds)

    plt.figure(figsize=(20, 6))

    plt.plot(selected_data['seconds'], selected_data['value'], label='Обучающие данные', color='blue')

    plt.plot(predicted_seconds + train_start, predictions_data, label='Предсказанные значения', color='orange')

    plt.plot(real_data_after_train['seconds'], real_data_after_train['value'],
             label='Реальные данные после периода обучения', color='green')

    plt.axvline(x=train_end, color='red', linestyle='--', label='Конец обучающего периода')

    plt.xlabel('Время (секунды с начала суток)')
    plt.ylabel('Значение')
    plt.title('График обучающих, реальных и предсказанных значений')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def make_predictions(df, start_second, train_time_seconds, model, scaler):
    data_to_predict = df.query('seconds >= @start_second and seconds <= @start_second + @train_time_seconds - 1')
    scaled_selected = scaler.transform(data_to_predict[['value']]).reshape((1, train_time_seconds, 1))

    predictions = scaler.inverse_transform(model.predict(scaled_selected))

    results_df = pd.DataFrame({
        'predictions': predictions.flatten(),
        'seconds': list(range(start_second + train_time_seconds, start_second + train_time_seconds + 21600))
    })

    results_df.to_excel('predictions.xlsx', index=False)

    return predictions


def plot_only_predictions(file_path):
    results_df = pd.read_excel(file_path)

    plt.figure(figsize=(20, 6))

    plt.plot(results_df['seconds'], results_df['predictions'], linestyle='-', color='b', linewidth=1.5)

    plt.title('Предсказания по секундам', fontsize=16)
    plt.xlabel('Секунды', fontsize=14)
    plt.ylabel('Предсказанные значения', fontsize=14)
    plt.grid()
    plt.xticks(rotation=45, fontsize=12)

    plt.ylim(-0.5, 8)

    plt.tight_layout()
    plt.show()