from data_preprocessing import read_txt_to_list

# read file to list
txt_path = "HSBC_Examples/TstUoB_2024-01-02LOBs.txt"
list_data = read_txt_to_list(txt_path)

print(list_data[10:20])
