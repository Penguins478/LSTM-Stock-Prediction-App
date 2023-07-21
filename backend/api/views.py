import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yfin
import io
import base64
from django.http import JsonResponse
from rest_framework.decorators import api_view

yfin.pdr_override()

matplotlib.use('agg')  # Use the 'agg' backend

# Load the pre-trained Keras model
model = load_model('/Users/ryan/Desktop/LSTM-Stock-Trading-App/keras_model.h5')

@api_view(['POST'])
def generate_prediction_graph(request):
    if request.method == 'POST':
        # stock_ticker = request.POST.get('stock_ticker', 'AAPL')
        # print(request.data['stock_ticker'])
        stock_ticker = request.data['stock_ticker']
        # print(stock_ticker)
        start = '2010-01-01'
        end = '2019-12-31'
        y_symbols = [stock_ticker]
        df = pdr.get_data_yahoo(y_symbols, start, end)

        ####################

        # Generate the three graphs
        plt.figure(figsize=(12, 18))

        # Graph 1: Closing Price vs. Time Chart
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
        plt.title('Closing Price vs. Time Chart')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        # Graph 2: Closing Price vs. Time Chart with Moving Averages
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
        plt.plot(df.index, ma100, label='100-day Moving Average', color='orange')
        plt.plot(df.index, ma200, label='200-day Moving Average', color='red')
        plt.title('Closing Price vs. Time Chart with Moving Averages')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7) : int(len(df))])


        scaler = MinMaxScaler(feature_range=(0,1))
        data_training_array = scaler.fit_transform(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)

        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor


        # Graph 3: Predictions vs. Original
        plt.subplot(3, 1, 3)
        plt.plot(y_test, label='Original Price', color='blue')
        plt.plot(y_predicted, label='Predicted Price', color='red')
        plt.title('Closing Price vs. Predicted Price')
        plt.xlabel('Time (Days)')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save the graphs as images
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        # Convert the images to base64 to send to the frontend
        graph_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Convert the DataFrame to a JSON object for the table
        # data_table = df.to_dict(orient='list')

        # Get the summary statistics using df.describe()
        data_summary = df.describe()

        # Convert the summary statistics to a dictionary for JSON response
        data_summary_dict = data_summary.to_dict()

        # print(graph_base64)

        return JsonResponse({
            'graph': graph_base64,
            'summary': data_summary_dict,
        })

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
