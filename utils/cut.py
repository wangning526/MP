import pandas as pd
import os
import multiprocessing as mp

def cut(filename):
    # 读取CSV文件
    df = pd.read_csv(filename,header=None)

    # 选择需要保存的行范围
    start_row = 54
    end_row = 129

    # 保存选定的行范围到新的DataFrame
    new_df = df.iloc[start_row:end_row]
    # 将新的DataFrame保存为CSV文件
    new_df.to_csv(filename, index=False,header=False)

if  __name__ == '__main__':
    pool = mp.Pool(processes=10)
    path = '../V10M'

    tasks = []
    for filename in os.listdir(path):
        filename1 = os.path.join(path, filename)

        task = pool.apply_async(cut, args=(filename1,))
        tasks.append(task)

    for task in tasks:
        task.get()

    pool.close()
    pool.join()

    print("All done!")