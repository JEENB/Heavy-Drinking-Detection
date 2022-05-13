import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

def plot_tac_reading(df):
	df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

	fig = go.Figure()
	
	for pid, group in df.groupby("pid"):

		fig.add_trace(go.Scatter(
			x=group["datetime"],
			y=group["TAC_Reading"],
			name = pid,
			mode = 'markers',
		))

	fig.add_hline(y=0.08)



	fig.update_layout(height=600, width=1200, title={
        'text': f"TAC Reading overtime for all user",
        'y':0.93,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
		xaxis_title = "TIme",
		
		yaxis_title = "TAC level"
		)
	fig.write_image(f"./plots/TAC_reading_all.png")


def get_acc_user(df, pid):
	'''returns all acceloremeter readings for a given pid'''
	return df.loc[(df["pid"]==pid) & (df["time"]!=0)].reset_index(drop=True)


### Plotting user cc6740's acceloremeter reading
def plot_acc_reading(df,pid):
	user_data = get_acc_user(df,pid)
	user_data['datetime'] = pd.to_datetime(user_data['time'], unit='ms')

	fig = make_subplots(rows=3, cols=1, x_title='Time',
                    y_title='Acceleration',)

	fig.append_trace(go.Scatter(
		x=user_data['datetime'],
		y=user_data["x"],
		name = "X-axis",
	), row=1, col=1)

	fig.append_trace(go.Scatter(
		x=user_data['datetime'],
		y=user_data["y"],
		name = "Y-axis",
	), row=2, col=1)

	fig.append_trace(go.Scatter(
		x=user_data['datetime'],
		y=user_data["z"],
		name = "Z-axis",
	), row=3, col=1)


	fig.update_layout(height=600, width=1200, title={
        'text': f"Accelerometer Reading for {pid}",
        'y':0.93,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
		)
    
	fig.show()

	if not os.path.exists('plots'):
		os.makedirs('plots')
    
	# fig.write_image(f"./plots/raw_acc_{pid}.png")


def preprocess_tac(path):
    """
	threshold = 0.08
    
    Convert "TAC_Reading" into binary "intoxicated" variable:
    intoxicated = 1 if TAC_Reading > 0.08,
    intoxicated = 0 if TAC_Reading <= 0.08.
	Returns concatenated dataframe with all pids.
    """
    appended_data = []
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        df = pd.read_csv(path + filename)
        df['pid'] = filename.split("_")[0]
        appended_data.append(df)
    df = pd.concat(appended_data).sort_values(['timestamp'], ascending=True).reset_index(drop=True)
    # Create binary flag.
    df.loc[df.TAC_Reading > 0.08, "intoxicated"] = 1
    df.loc[df.TAC_Reading <= 0.08, "intoxicated"] = 0
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    return df

def missing_data_imputation(df):
    """
    Standardize: generate max_timestamp (ms) - min_timestamp (ms) 
    number of rows for a given df, so that there is a row for 
    every millisecond. 
    Impute: Fill in missing accelerometer readings with readings from the 
    previous millisecond timestamp.
    Returns: array of timestamps and array of accelerometer readings.
    """
    min_timestamp = df['time'].min()
    max_timestamp = df['time'].max()
    min_timeinterval = 1

    print(min_timestamp, max_timestamp)

    array_size = int((max_timestamp-min_timestamp)/min_timeinterval) + 1
    # Initialize empty array of size array_size.
    accelerometer_readings = [None] * array_size

    # Add data to the arrays based on the readings.
    first_accelerometer_reading = None
    for i in range(0, len(df)):
        if(first_accelerometer_reading == None):
            first_accelerometer_reading = [
                df.loc[i, 'x'], df.loc[i, 'y'], df.loc[i, 'z']]
        index = int((df.loc[i, 'time'] - min_timestamp)/min_timeinterval)
        try:
            accelerometer_readings[index] = [
                df.loc[i, 'x'], df.loc[i, 'y'], df.loc[i, 'z']]
        except:  # If sensor readings are empty. -- Erroroneous.
            pass

    prev_accelerometer = None
    for i in range(0, array_size):
        # If missing, add reading from previous timestamp.
        if(accelerometer_readings[i] == None):
            if(prev_accelerometer != None):
                accelerometer_readings[i] = prev_accelerometer
            else:
                accelerometer_readings[i] = first_accelerometer_reading
        # If not missing, skip row and do not override it.
        elif (accelerometer_readings[i] != None):
            prev_accelerometer = accelerometer_readings[i]

    return list(range(min_timestamp, max_timestamp+1)), accelerometer_readings

def preprocess_acc(path, new_path):
    """
    Given a path load "all_accelerometer_data_pids_13.csv" 
    and drop if missing time or zero accelerometer data.
    For each pid, standardize sampling frequency to every millisecond, 
    impute missing data and save a pickle file for each pid to new_path.
    This function returns None.
    """
    # Create folder for new_path if does not exist.
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # Load file.
    df = pd.read_csv(path + "all_accelerometer_data_pids_13.csv")
    # Drop if missing timestamp.
    df = df.loc[df.time!=0]
    # Keep non-zero accelerometer data only.
    df = df.loc[(df.x!=0) & (df.y!=0) & (df.z!=0)]
    # For each pid, standardize sampling frequency to 1 millisecond.
    for current_pid in [
# "CC6740",
# "DC6359",
# "DK3500",
# "HV0618",
# "JB3156",
# "JR8022",
# "MC7070",
# "MJ8002",
# "PC6771",
"SF3079"
]:
        print(f"Preprocessing: {current_pid}")
        temp = df.loc[df.pid == current_pid].sort_values('time', ascending=True).reset_index(drop=True)
        print(f"Original shape: {temp.shape}")
        timestamps, readings = missing_data_imputation(temp)
# Create df with timestamps and readings.
        new_df = pd.DataFrame(readings, columns=["x", "y", "z"])
        new_df['time'] = timestamps
        new_df['pid'] = current_pid
        # Print new df shape.
        print(f"New shape: {new_df.shape}")
        # Export preproccessed data as a pickle file.
        new_df.to_pickle(new_path + current_pid + 
        "_preprocessed_acc.pkl")
        print("Preprocessing complete and files exported.")
    return None


# # Pre-process accelerometer data for each pid and save as pkl file.
# path = "data/"
# new_path = "preprocessed/"
# preprocess_acc(path, new_path)


