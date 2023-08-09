import numpy as np

extract_path = './NewData/xyzProj/Test_'
sub_dirs=['Circle', 'DiagonalLeftToRight', 'DiagonalRightToLeft', 'FistForward', 'FistSpread', 'LeftRight', 'UpToDown']

def read_datasetFile(labelname):
    Data_path = extract_path+labelname
    data = np.load(Data_path+'.npz')
    return data['arr_0'], data['arr_1']

min_data_len = 999999
datas = []
for dir in sub_dirs:
    data, label = read_datasetFile(dir)
    datas.append([data, label])
    len = data.shape[0]
    print(len)
    if len < min_data_len:
        min_data_len = len
"""
for data, label in datas:
    print(data[:min_data_len].shape)
    np.savez(extract_path, data[:min_data_len], label[:min_data_len])
    del data, label

"""