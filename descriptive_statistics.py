import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

file_path = 'data/merged_data.csv'
df = pd.read_csv(file_path)

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df['date'] = df['timestamp'].dt.date

df_grouped = df.groupby(['sensor_id', 'date'])['value'].mean().reset_index()
sensor_ids = df_grouped['sensor_id'].unique()
sensor_ids = sensor_ids[:18]

plt.figure(figsize=(12, 6))

for sensor_id in sensor_ids:
    df_sensor = df_grouped[df_grouped['sensor_id'] == sensor_id]
    plt.plot(df_sensor['date'], df_sensor['value'], marker='o', linestyle='-', color='black', label=f'sensor_id {sensor_id}')

plt.axhline(y=400, color='red', linestyle='--', label='Upper Limit')
plt.axhline(y=200, color='blue', linestyle='--', label='Lower Limit')

plt.title('')
plt.xlabel('', fontsize=14)
plt.ylabel('Tensiometer Value (mbar)', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

legend_elements = [
    Line2D([0], [0], color='red', linestyle='--', label='Upper Limit'),  # Red dashed line
    Line2D([0], [0], color='blue', linestyle='--', label='Lower Limit'),  # Blue dashed line
    Line2D([0], [0], color='black', marker='o', linestyle='-', label='Generic Tensiometer')  # Black solid line with markers
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('images/time_series_tens.png')
plt.show()


df_descr = df.groupby('sensor_id')['value'].agg(['mean', 'median', 'min', 'max', 'std'])
df_descr = df_descr.round(2).sort_values(by='mean')
df_descr = df_descr.reset_index(drop=True)
df_descr.index = df_descr.index + 1
df_descr.index.name = 'Sensor ID'

fig, ax = plt.subplots(1, 1)
ax.axis('off')

table = plt.table(cellText=df_descr.values, colLabels=df_descr.columns, rowLabels=df_descr.index, cellLoc = 'center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

latex_table = df_descr.to_latex(index=True, column_format='|c|c|c|c|c|c|', bold_rows=True)
with open('descriptive_statistics.tex', 'w') as f:
    f.write(latex_table)

plt.savefig('images/descriptive_statistics.png')
