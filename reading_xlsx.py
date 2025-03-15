import os
import numpy as np
import pandas as pd

def read_xy():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'data.xlsx')
    
    # 读取excel中指定的两个表单，默认使用第一行为列名
    sheets = pd.read_excel(file_path, sheet_name=['train', 'test'])
    
    train_data = sheets['train'].to_numpy()
    test_data = sheets['test'].to_numpy()

    # print(train_data.shape)
    # print(test_data.shape)
    return train_data[:,0], train_data[:,1], test_data[:,0], test_data[:,1] 

# 示例调用
if __name__ == '__main__':
    train_x,train_y,test_x,test_y = read_xy()
    #print(data_array.shape)
    #print(data_array)