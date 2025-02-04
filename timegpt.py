import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from nixtla import NixtlaClient
import time
import logging

# Initialize logging
logging.basicConfig(filename='execution_log.txt', level=logging.INFO)
logging.getLogger('nixtla.nixtla_client').setLevel(logging.WARNING)
# Initialize TimeGPT client
client = NixtlaClient(api_key='YOUR_API_KEY')

dataset_folder = "your_dataset.csv"

# Load and prepare data
df = pd.read_csv(dataset_folder)
df = df.rename(columns={'date_column': 'ds', 'target_column_name': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

# Split data into train and test
train_size = int(len(df) * 0.85)
train, test = df[:train_size], df[train_size:].reset_index(drop=True)

# Parameters
HORIZON = 6
MAX_RETRIES = 3
BATCH_SIZE = 20  # Number of rows to process in each batch
output_csv_file = 'test_predictions_TimeGPT.csv'

# Start measuring execution time
start_time = time.time()

# Initialize storage for predictions
rolling_predictions = []

# Iterate through the test dataset in batches
for batch_start in range(0, len(test) - HORIZON + 1, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(test) - HORIZON + 1)
    batch_predictions = []

    for i in range(batch_start, batch_end):
        # Exclude 'TimeGPT_predicted' from the future window if it exists
        future_window = test.drop(columns=['y', 'TimeGPT_predicted'], errors='ignore').iloc[i:i + HORIZON].copy()

        retries = 0
        while retries < MAX_RETRIES:
            try:
                # Forecast using TimeGPT
                forecast_result = client.forecast(
                    df=train,
                    X_df=future_window,
                    h=HORIZON,
                    freq='h',
                    id_col='unique_id',
                    time_col='ds',
                    target_col='y'
                )

                # Store the first prediction from this window
                batch_predictions.append(forecast_result['TimeGPT'].iloc[0])

                logging.info(f"Iteration {i + 1}/{len(test) - HORIZON + 1} completed successfully.")
                break  # Exit retry loop on success
            except Exception as e:
                retries += 1
                logging.error(f"Iteration {i + 1}: Attempt {retries} failed with error: {e}")
                if retries >= MAX_RETRIES:
                    logging.error(f"Iteration {i + 1}: Failed after {MAX_RETRIES} retries.")
                    batch_predictions.append(np.nan)  # Append NaN if prediction fails


    # Append batch predictions to the overall list
    rolling_predictions.extend(batch_predictions)

    # Save intermediate results to CSV after each batch
    test['TimeGPT_predicted'] = pd.Series(rolling_predictions)
    test[['ds', 'y', 'TimeGPT_predicted']].to_csv(output_csv_file, index=False)

    logging.info(f"Batch {batch_start // BATCH_SIZE + 1} saved to {output_csv_file}.")

# Stop measuring execution time
execution_time = time.time() - start_time

# Calculate evaluation metrics
valid_rows = ~test['TimeGPT_predicted'].isna()
mae = mean_absolute_error(test['y'][valid_rows], test['TimeGPT_predicted'][valid_rows])
mse = mean_squared_error(test['y'][valid_rows], test['TimeGPT_predicted'][valid_rows])
rmse = np.sqrt(mse)
r2 = r2_score(test['y'][valid_rows], test['TimeGPT_predicted'][valid_rows])

# Log and print metrics
metrics_log = (f"Execution Time: {execution_time:.2f} seconds\n"
               f"MAE: {mae:.2f}\n"
               f"MSE: {mse:.2f}\n"
               f"RMSE: {rmse:.2f}\n"
               f"R²: {r2:.2f}\n")

print(metrics_log)
logging.info(metrics_log)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(test['ds'][valid_rows], test['y'][valid_rows], label='Actual Boarding Count')
plt.plot(test['ds'][valid_rows], test['TimeGPT_predicted'][valid_rows], label='Predicted Boarding Count', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Boarding Count')
plt.title('Actual vs Predicted Boarding Count (6-Hour Ahead Forecast)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the plot
figure_file_path = 'forecast_results.png'
plt.savefig(figure_file_path, dpi=600)
plt.show()