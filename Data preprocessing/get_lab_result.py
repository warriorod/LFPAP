import pandas as pd

ica_list = []
tbil_list = []
wbc_list = []
ptsec_list = []
df = pd.read_csv('../vitaldb/clinical_data.csv',usecols=[0,6])
lab_df = pd.read_csv('../vitaldb/lab_data.csv')
first_column_values = df.values[:,0]
second_column_values = df.values[:, 1]
feature_list = ['ica','tbil','wbc','ptsec']
for i in range(first_column_values.shape[0]):
    index = first_column_values[i]
    t = second_column_values[i]
    for feature in feature_list:
        filtered_df = lab_df[(lab_df['caseid'] == index)& (lab_df['name'] == feature)]
        if filtered_df.empty:
            print("No features matching the criteria were found.")
            closest_value = -1
        else:
            filtered_df = filtered_df[filtered_df['dt'] < t]
            if filtered_df.empty:
                print("No features matching the criteria were found.")
                closest_value = -1
            else:
                closest_row = filtered_df.sort_values(by='dt', ascending=False).iloc[0]
                closest_value = closest_row['result']
        if feature == 'ica':
            ica_list.append(closest_value)
        elif feature == 'tbil':
            tbil_list.append(closest_value)
        elif feature == 'wbc':
            wbc_list.append(closest_value)
        elif feature == 'ptsec':
            ptsec_list.append(closest_value)

data = {
    'ica':ica_list,
    'tbil':tbil_list,
    'wbc':wbc_list,
    'ptsec':ptsec_list
}

df_save = pd.DataFrame(data)

df_save.to_csv('lab_for_test.csv',encoding='utf-8', index=False)