import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from package.idw import holdout_idw, kfoldcv_idw
from package.kriging import holdout_kriging, kfoldcv_kriging


readings = pd.read_csv("data/merged_data.csv")
boundary = pd.read_csv("data/boundary_tres.csv")

rmse_results1 = holdout_idw(readings, boundary, 100, 2, 0.8)
rmse_results2 = holdout_kriging(readings, boundary, 100, 100, 0.8)
rmse_results3 = kfoldcv_idw(readings, boundary, 100, 2, 6)
rmse_results4 = kfoldcv_kriging(readings, boundary, 100, 10, 6)
rmse_results5 = kfoldcv_idw(readings, boundary, 100, 2, 18)
rmse_results6 = kfoldcv_kriging(readings, boundary, 100, 10, 18)


dataframes = [pd.DataFrame(list(results.items()), columns=['timestamp', 'rmse']) for results in
              [rmse_results1, rmse_results2, rmse_results3, rmse_results4, rmse_results5, rmse_results6]]
titles = ['Holdout IDW', 'Holdout Kriging', 'K-fold IDW', 'K-fold Kriging', 'LOOCV IDW', 'LOOCV Kriging']

df_list = []

for df, method in zip(dataframes, titles):
    df['method'] = method
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df[['Validation Technique', 'Interpolation Method']] = combined_df['method'].str.split(' ', n=1, expand=True)

combined_df.drop(columns=['method'], inplace=True)
combined_df.rename(columns={'rmse': 'RMSE'}, inplace=True)

sns.boxplot(data=combined_df, x="Validation Technique", y="RMSE", hue="Interpolation Method", gap=0.1, showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7})

plt.gca().set_facecolor('white')

plt.tight_layout()
plt.savefig('images/rmse_comparison_boxplots.png')
