import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

raw_data = pd.read_excel('Volve production data.xlsx')

raw_data = raw_data[['DATEPRD', 'NPD_WELL_BORE_NAME', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
                     'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P', 'BORE_OIL_VOL',
                     'BORE_GAS_VOL', 'BORE_WAT_VOL']]

F14 = pd.DataFrame(raw_data[raw_data['NPD_WELL_BORE_NAME'] == '15/9-F-14'])

# F14 oil rate before pre-processing
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(F14.DATEPRD, F14.BORE_OIL_VOL, color='tab:blue', label='Oil Rate')
ax.set_xlabel('Time')
ax.set_ylabel('Oil Rate')
ax.set_title('Well F14')
ax.grid(True)
ax.legend(loc='upper right')

# F14 gas rate before pre-processing
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(F14.DATEPRD, F14.BORE_GAS_VOL, color='tab:blue', label='Gas Rate')
ax.set_xlabel('Time')
ax.set_ylabel('Gas Rate')
ax.set_title('Well F14')
ax.grid(True)
ax.legend(loc='upper right')

# F14 statistics before pre-processing
preF14describe = F14.describe()
preF14describe.to_excel('Volve_well/preF14describe.xlsx')

# Downhole pressure analyzing
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=F14["AVG_DOWNHOLE_PRESSURE"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(F14["AVG_DOWNHOLE_PRESSURE"])

# Bore oil volume analyzing
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=F14["BORE_OIL_VOL"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(F14["BORE_OIL_VOL"])

# Bore gas volume analyzing
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=F14["BORE_GAS_VOL"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(F14["BORE_GAS_VOL"])

# Bore water volume analyzing
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=F14["BORE_WAT_VOL"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(F14["BORE_WAT_VOL"])

# Bore avg choke size analyzing
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=F14["AVG_CHOKE_SIZE_P"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(F14["AVG_CHOKE_SIZE_P"])

# Bore avg whp analyzing
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=F14["AVG_WHP_P"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(F14["AVG_WHP_P"])

F14.drop(columns='NPD_WELL_BORE_NAME').replace(0, np.nan) .dropna()

# Rename features
F14 = F14.rename(columns={'DATEPRD': 'DATE', 'AVG_DOWNHOLE_PRESSURE': 'BTHP', 'AVG_DOWNHOLE_TEMPERATURE': 'BTHT',
                          'AVG_DP_TUBING': 'DP', 'AVG_CHOKE_SIZE_P': 'CS', 'AVG_WHP_P': 'WHP', 'AVG_WHT_P': 'WHT',
                          'BORE_OIL_VOL': 'OIL', 'BORE_GAS_VOL': 'GAS', 'BORE_WAT_VOL': 'WATER'})

# Create new feature (time step)
F14.insert(1, 'STEP', 0)
for i in range(1, len(F14['DATE'])):
    F14.iloc[i, 1] = pd.Timedelta(F14.iloc[i, 0] - F14.iloc[0, 0]).days

F14.to_csv('Volve_well/F14_raw.csv')
F14.to_excel('Volve_well/F14_raw.xlsx')
F14_GAS_raw = F14[['DATE', 'STEP', 'GAS']]
F14_GAS_raw.to_csv('Volve_well/F14_GAS_raw.csv')
F14_GAS_raw.to_excel('Volve_well/F14_GAS_raw.xlsx')

# Remove F14's missing values or outliers
F14.drop(F14.loc[F14['BTHP'] < 100].index, inplace=True)
F14.drop(F14.loc[F14['OIL'] < 100].index, inplace=True)
F14.drop(F14.loc[F14['GAS'] < 100].index, inplace=True)
F14.drop(F14.loc[F14['WATER'] < 100].index, inplace=True)
F14.drop(F14.loc[F14['CS'] < 20].index, inplace=True)
F14.drop(F14.loc[F14['WHP'] < 10].index, inplace=True)
F14.reset_index(inplace=True)
F14.drop(columns='index', inplace=True)

# F14 statistics after pre-processing
postF14describe = F14.describe()
postF14describe.to_excel('Volve_well/postF14describe.xlsx')

# Save data for ML/DL aim
F14.to_csv('Volve_well/F14.csv')
F14.to_excel('Volve_well/F14.xlsx')
F14_gas = F14[['DATE', 'STEP', 'GAS']]
F14_gas.to_csv('Volve_well/F14_GAS.csv')
F14_gas.to_excel('Volve_well/F14_GAS.xlsx')

# F14 oil rate after pre-processing
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(F14.DATE, F14.OIL, color='tab:blue', label='Oil Rate')
ax.set_xlabel('Time')
ax.set_ylabel('Oil Rate')
ax.set_title('Well F14')
ax.grid(True)
ax.legend(loc='upper right')

# F14 gas rate after pre-processing
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(F14.DATE, F14.GAS, color='tab:blue', label='Gas Rate')
ax.set_xlabel('Time')
ax.set_ylabel('Gas Rate')
ax.set_title('Well F14')
ax.grid(True)
ax.legend(loc='upper right')

# F14 Correlation Matrix
correlation = F14.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
mask = mask[1:, :-1]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data=correlation.iloc[1:, :-1], mask=mask, annot=True, fmt=".2f",
            cmap="crest", vmin=-1, vmax=1, linecolor="white", linewidths=0.5)
title = "CORRELATION MATRIX"
ax.set_title(title, loc='center', fontsize=15)

plt.show()
