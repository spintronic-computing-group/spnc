import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import MultipleLocator
from scipy.ndimage import gaussian_filter1d
import csv
from scipy.interpolate import interp1d
import shutil

import datetime
import re
from scipy.optimize import curve_fit
import pandas as pd
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def pickup(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        x = np.array(data["coil_1_field"])
        y = np.array(data["kerr_signal"])
    except json.JSONDecodeError:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            x, y = [], []
            for row in reader:
                try:
                    x_val = float(row[0])
                    y_val = float(row[1])
                    x.append(x_val)
                    y.append(y_val)
                except ValueError:
                    continue  
        x = np.array(x)
        y = np.array(y)


    if np.isnan(x).any() or np.isnan(y).any():
        mask = ~np.isnan(x) & ~np.isnan(y)  
        x = x[mask]
        y = y[mask]

    x_min = x.min()
    x_max = x.max() 


    mask = (x >= x_min) & (x <= x_max)
    x_cleaned = x[mask]
    y_cleaned = y[mask]

    return x_cleaned, y_cleaned

def plot_single(filepath, base_save_path, title=None, xlabel='Integrated Field Average', ylabel='Normalised Magnetisation', figsize=(10, 6), show=True):

    filename = os.path.splitext(os.path.basename(filepath))[0]
    save_path = os.path.join(base_save_path, f'{filename}.png')



    match = re.search(r'_(\d+)$', filename)
    suffix = match.group(1) if match else None
    match = re.search(r'^[A-Z]+\d+', filename)
    prefix = match.group() if match else None


    plt.figure(figsize=figsize)
    x, y = pickup(filepath)

    plt.plot(x, y)

    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MultipleLocator(50))
    plt.grid(True)
    plt.savefig(save_path)
    if show:
        plt.show()
    # plt.close()
    return 

path = 'G:\\Shared drives\\FMM\\Data\\FMOKE\\Chen\\Zurich Sample_organised\\New folder\\'
path = 'C:\\Users\\Chen\\Desktop\\Repository\\spnc'

plot_single(path,path)
