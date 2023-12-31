import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import matplotlib.colors as colors
import pandas as pd
import numpy as np

def gen_grid(mp, savename):
    # 读取原始数据CSV文件
    res_grid = np.zeros([75 * 360, 3])
    index = 0
    lat = 37
    for i in range(mp.shape[0]):
        for j in range(mp.shape[1]):
            res_grid[index][0] = (j + 1)
            res_grid[index][1] = lat
            res_grid[index][2] = mp.iloc[i][j]
            index += 1
        lat -= 1
    df = pd.DataFrame(res_grid)
    df.to_csv(savename, index=False, header=False)
    print('save grid data to ' + savename)


def gen_polt(data, savename):
    data = np.array(data)
    # 将空格替换为 0
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
    ax.set_extent([-180, 180, -90, 90])

    # 绘制地图数据点
    ax.scatter(longitude, latitude, c=value, cmap='jet', edgecolors='none',
               alpha=0.3, s=1, transform=ccrs.PlateCarree())

    # 绘制地图边界和海岸线
    ax.coastlines()

    # 添加经纬度网格线
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')

    # 设置颜色映射和归一化范围
    vmin = np.min(0)
    vmax = np.max(7.0e+12)
    print('min = ' + str(vmin) + ' max = ' + str(vmax))
    norm = colors.Normalize(vmin=0, vmax=7.0e+12)
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
    # plt.show()







