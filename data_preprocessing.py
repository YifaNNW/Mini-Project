import pandas as pd


def read_file(txt_path):
    with open(txt_path, 'r') as f:
        data = []
        for line in f.readlines():
            line = line.strip()
            if line:
                data.append(line)
    return data


def read_txt_to_list(txt_path):
    txt_data = read_file(txt_path)
    list_data = []
    # Convert string to list
    for each_str in txt_data:
        list_data.append(eval(each_str))
    return list_data


def lob_time(lob):
    return lob[1]


def get_lobs_by_tapes(df_tapes, list_lobs):
    timestamp = df_tapes['timestamp']
    list_timestamp = timestamp.tolist()
    # Select the LOBs that are in the same second as the tape
    lobs_selected = []
    idx_lobs = 0
    for each_timestamp in list_timestamp:
        while lob_time(list_lobs[idx_lobs]) < each_timestamp:
            idx_lobs += 1
        if lob_time(list_lobs[idx_lobs]) == each_timestamp:
            lobs_selected.append(list_lobs[idx_lobs])
    return lobs_selected


def preprocess_tapes(csv_path):
    # Read csv file
    data = pd.read_csv(csv_path)
    # Remove the first two columns
    data.drop(data.columns[[0, 1]], axis=1, inplace=True)
    # Name the columns
    data.columns = ['timestamp', 'price', 'volume']
    return data


txt_path = "HSBC_Examples/TstUoB_2024-01-02LOBs.txt"
lobs = read_txt_to_list(txt_path)

csv_path = "HSBC_Examples/TstUoB_2024-01-02tapes.csv"
tapes = preprocess_tapes(csv_path)

# Get the LOBs that are in the same second as the tape
lobs_selected = get_lobs_by_tapes(tapes, lobs)

# Save the selected LOBs to a csv file
df_lobs_selected = pd.DataFrame(lobs_selected)
df_lobs_selected.to_csv("HSBC_Examples/TstUoB_2024-01-02LOBs_Selected.csv", index=False)








