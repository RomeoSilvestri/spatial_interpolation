import pandas as pd
import matplotlib.pyplot as plt

from package.idw import kfoldcv_idw
from package.kriging import kfoldcv_kriging


readings = pd.read_csv("data/merged_data.csv")
boundary = pd.read_csv("data/boundary_tres.csv")

rmse_results1 = kfoldcv_idw(readings, boundary, 100, 2, 6)
rmse_results2 = kfoldcv_kriging(readings, boundary, 100, 10, 6)

rmse_df_idw = pd.DataFrame(list(rmse_results1.items()), columns=['timestamp', 'rmse_idw'])
rmse_df_kriging = pd.DataFrame(list(rmse_results2.items()), columns=['timestamp', 'rmse_kriging'])

rmse_df_idw['timestamp'] = pd.to_datetime(rmse_df_idw['timestamp'])
rmse_df_kriging['timestamp'] = pd.to_datetime(rmse_df_kriging['timestamp'])

merged_rmse = pd.merge(rmse_df_idw, rmse_df_kriging, on='timestamp', how='outer')
daily_rmse = merged_rmse.resample('D', on='timestamp').mean().reset_index()
daily_rmse['timestamp'] = daily_rmse['timestamp'].dt.date

plt.figure(figsize=(12, 6))

plt.plot(daily_rmse['timestamp'], daily_rmse['rmse_idw'], marker='o', linestyle='-',
         label='Inverse Distance Weighting', color='blue')

plt.plot(daily_rmse['timestamp'], daily_rmse['rmse_kriging'], marker='o', linestyle='-',
         label='Ordinary Kriging', color='orange')

plt.title('')
plt.xlabel('', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('', fontsize=14)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('images/rmse_comparison_kfold.png')
