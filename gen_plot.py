import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter
import matplotlib.colors as colors
import multiprocessing as mp
import os



def mp_polt(file_name,savename):
    data = pd.read_csv(file_name, header=None)
    data = np.array(data)

    # 应用对数函数
    data = data.round(2)
    df = pd.DataFrame(data)
    # 替换第三列的NaN和inf为0
    df.iloc[:, 2] = df.iloc[:, 2].replace([np.nan, np.inf, -np.inf], 0)
    data = df
    data.iloc[:, 2] = data.iloc[:, 2].apply(lambda x: 0 if x < 0 else x)
    data.iloc[:, 2] = data.iloc[:, 2].apply(lambda x: 0 if x > 7.0e+12 else x)
    # 删除第三列数值为0的行
    data = data[data.iloc[:, 2] != 0]
    data = np.array(data)

    longitude = data[:, 0]
    latitude = data[:, 1]
    value = data[:, 2]

    # 创建地图

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([-180,180,-90,90])

    # 绘制地图数据点
    ax.scatter(longitude, latitude, c=value, cmap='jet', edgecolors='none',
                    alpha=0.3,s=1, transform=ccrs.PlateCarree())

    # 绘制地图边界和海岸线
    ax.coastlines()

    # 添加经纬度网格线
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')

    # 设置颜色映射和归一化范围
    vmin = np.min(value)
    vmax = np.max(value)
    print(vmin, vmax)
    norm = colors.Normalize(vmin=-10, vmax=10)
    cmap = plt.get_cmap('jet')

    # 创建ScalarMappable对象并设置归一化范围和颜色映射
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 设置一个空数组以确保颜色映射正确应用于colorbar

    # 设置颜色图例
    cbar = plt.colorbar(sm, format=LogFormatter(), extend='neither', shrink=0.8)
    cbar.set_label('Value')

    # 保存图像
    plt.savefig(savename, dpi=600, bbox_inches='tight', pad_inches=0)
    print(savename + ' saved')
    # 显示图形
    #plt.show()

if __name__ == '__main__':
    path_mp = 'result'
    spath = 'plot'
    if not os.path.exists(spath):
        os.mkdir(spath)
    pool = mp.Pool(processes=6)
    tasks = []
    for file_name in os.listdir(path_mp):
        file_name = os.path.join(path_mp, file_name)
        savename = spath + '_' + file_name[-12:-4] + '.png'
        savename = os.path.join(spath, savename)
        task = pool.apply_async(mp_polt, args=(file_name, savename))
        tasks.append(task)
    for task in tasks:
        task.get()
    pool.close()
    pool.join()








