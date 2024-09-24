import pandas as pd


df_clinical = pd.read_csv('clinical_data.csv')
df_lab = pd.read_csv('lab.csv')
df_cr = df_lab[df_lab['name'] == 'cr']

df_clinical['AKI'] = 0


for index, row in df_clinical.iterrows():

    same_caseid_rows = df_cr[df_cr['caseid'] == row['caseid']]

    time_condition = (same_caseid_rows['dt'] > row['opend']) & (same_caseid_rows['dt'] <= row['opend'] + 604800)
    filtered_rows = same_caseid_rows[time_condition]


    if filtered_rows.empty:
        df_clinical.at[index, 'AKI'] = 'null'
    if (filtered_rows['result'] > row['preop_cr'] * 1.5).any():
        df_clinical.at[index, 'AKI'] = 1


df_clinical.to_csv('aki_7d.csv', index=False)
