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


txt_path = "HSBC_Examples/TstUoB_2024-01-02LOBs.txt"
list_data = read_txt_to_list(txt_path)

print(list_data[10:20])






