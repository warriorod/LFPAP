import numpy as np
import vitaldb
import pandas as pd

def count_points_in_range(arr, lower_bound, upper_bound):
    """
    Find and count the points in an array that fall within a specific range.
    
    Parameters:
    arr (numpy array): Input array
    lower_bound (float): Lower bound of the range
    upper_bound (float): Upper bound of the range
    
    Returns:
    int: The number of points within the range
    """
    # Create a boolean mask to mark points within the range
    mask = (arr >= lower_bound) & (arr <= upper_bound)
    
    # Count the number of points within the range
    count = np.sum(mask)
    
    return count

MBP_list = []
HR_list = []
SPO2_list = []
BT_list = []
df = pd.read_csv('../vitaldb/clinical_data.csv',usecols=[0])
first_column_values = df.iloc[:, 0]
for i in first_column_values:
    # print(i)
    vals = vitaldb.load_case(i, ['Solar8000/ART_MBP','Solar8000/HR','Solar8000/PLETH_SPO2','Solar8000/BT'], 15)
    MBP = vals[:,0]
    HR = vals[:,1]
    SPO2 = vals[:,2]
    BT = vals[:,3]
    LMBP = count_points_in_range(MBP,50,60) * 0.25
    HHR = count_points_in_range(HR,100,180) * 0.25
    LSPO2 = count_points_in_range(SPO2,70,90) * 0.25
    LBT = count_points_in_range(BT,34,36) * 0.25
    MBP_list.append(LMBP)
    HR_list.append(HHR)
    SPO2_list.append(LSPO2)
    BT_list.append(LBT)

data = {
    'MBP':MBP_list,
    'HR':HR_list,
    'SPO2':SPO2_list,
    'BT':BT_list
}

df_save = pd.DataFrame(data)

df_save.to_csv('interop.csv',encoding='utf-8', index=False)