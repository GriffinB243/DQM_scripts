# %matplotlib ipympl
# By Oris Salviano Neto, March 2026
# Simple event plotter for the pSCT


import numpy as np
# from scipy.optimize import curve_fit
import pandas as pd
import target_io

import matplotlib as mpl
# mpl.use('widget')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import PercentFormatter
import matplotlib.animation as animation
import glob

# import seaborn as sns
import argparse

# import os
import csv

pxs_p_quad = 16
quads_p_module = 4

fpms_t = ['7-21', '7-22', '7-23', '7-24', '7-15', '7-16', '7-17', '7-18', '7-19', '7-10', '7-11', '7-12', '7-13', '7-14', '7-5', '7-6', '7-7', '7-8', '7-9', '7-0', '7-1', '7-20', '7-2', '7-3', '7-4']  # Hardcoded. Can probably be automated by reading the active slots in the reader object?
displaay = False

modes_of_images = ["peak", "time", "ani", "img", "charge", "std"]

def load_config(filepath):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
    return result

config_properties_base = {"module_config_path": "./module_config.csv",
                        "run_file_path": "/data/wipac/CTA/targetcdata/run{0}_subrun{1}_r*.tio",
                        "pixel_grid": "True",
                        "module_grid": "True",
                        "sector_grid": "True",
                        "default_display": "sector",
                        "default_image": "peak",
                        "empty_modules_nans": "False",
                        "title_font": "15",
                        "colorbar_legend_font": "15",
                        "colorbar_ticks_font": "15",
                        "time_legend_font": "8",
                        "image_size": "6",
                        "colorscheme": "viridis",
                        "low_bound" : "None",
                        "high_bound" : "None",
                        "charge_window": "5",
                        "charge_baseline_subtraction": "False"
                        }

def load_config_objects():
    global config_properties
    try:
        config_properties = load_config("./config.txt")
    except:
        print("No config file found. Loading standards")
        config_properties = config_properties_base

    ## LOADING PROPERTIES
    global make_nans
    try:
        make_nans = config_properties["empty_modules_nans"] == "True"
    except:
        make_nans = config_properties_base["empty_modules_nans"] == "True"

    global pix_gridd
    try:
        pix_gridd = config_properties["pixel_grid"] == "True"
    except:
        pix_gridd = config_properties_base["pixel_grid"] == "True"

    global mod_gridd
    try:
        mod_gridd = config_properties["module_grid"] == "True"
    except:
        mod_gridd = config_properties_base["module_grid"] == "True"

    global sector_gridd
    try:
        sector_gridd = config_properties["sector_grid"] == "True"
    except:
        sector_gridd = config_properties_base["sector_grid"] == "True"

    global default_display
    try:
        default_display = config_properties["default_display"]
    except:
        default_display = config_properties_base["default_display"]

    global default_image
    try:
        default_image = config_properties["default_image"]
    except:
        default_image = config_properties_base["default_image"]

    global title_font
    try:
        title_font = float(config_properties["title_font"])
    except:
        title_font = float(config_properties_base["title_font"])

    global colorbar_legend_font
    try:
        colorbar_legend_font = float(config_properties["colorbar_legend_font"])
    except:
        colorbar_legend_font = float(config_properties_base["colorbar_legend_font"])

    global colorbar_ticks_font
    try:
        colorbar_ticks_font = float(config_properties["colorbar_ticks_font"])
    except:
        colorbar_ticks_font = float(config_properties_base["colorbar_ticks_font"])

    global time_legend_font
    try:
        time_legend_font = float(config_properties["time_legend_font"])
    except:
        time_legend_font = float(config_properties_base["time_legend_font"])

    global image_size
    try:
        image_size = float(config_properties["image_size"])
    except:
        image_size = float(config_properties_base["image_size"])

    global path_to_file_def
    try:
        path_to_file_def = config_properties["run_file_path"] #.format(run, subrun)
    except:
        path_to_file_def = config_properties_base["run_file_path"] #.format(run, subrun)

    global colorscheme_default
    try:
        colorscheme_default = config_properties["colorscheme"]
    except:
        colorscheme_default = config_properties_base["colorscheme"]

    global low_bound
    try:
        low_bound = config_properties["low_bound"]
    except:
        low_bound  = config_properties_base["low_bound"]

    try:
        low_bound = float(low_bound)
    except:
        low_bound = None

    global high_bound
    try:
        high_bound = config_properties["high_bound"]
    except:
        high_bound  = config_properties_base["high_bound"]

    try:
        high_bound = float(high_bound)
    except:
        high_bound = None

    global charge_window
    try:
        charge_window = config_properties["charge_window"]
    except:
        charge_window  = config_properties_base["charge_window"]

    try:
        charge_window = int(charge_window)
    except:
        charge_window  = int(config_properties_base["charge_window"])
    global charge_baseline_subtraction
    try:
        charge_baseline_subtraction = config_properties["charge_baseline_subtraction"] == "True"
    except:
        charge_baseline_subtraction = config_properties_base["charge_baseline_subtraction"] == "True"

    global charge_low_bound
    global charge_high_bound
    global img_argss
    charge_low_bound = None
    charge_high_bound = None
    if low_bound != None:
        charge_low_bound = charge_window*low_bound
    if high_bound != None:
        charge_high_bound = charge_window*high_bound
    img_argss =     {"peak": [image_size, pix_gridd, mod_gridd, sector_gridd, low_bound, high_bound],
                    "time": [image_size, pix_gridd, mod_gridd, sector_gridd, None, None],
                    "std": [image_size, pix_gridd, mod_gridd, sector_gridd, None, None],
                    "ani":  [image_size, pix_gridd, mod_gridd, sector_gridd, low_bound, high_bound],
                    "img":  [image_size, pix_gridd, mod_gridd, sector_gridd, low_bound, high_bound],
                    "charge":  [image_size, pix_gridd, mod_gridd, sector_gridd, charge_low_bound, charge_high_bound],
                    }

load_config_objects()

colorbar_desc = {"peak": "Peak amplitude [ADC value]",
                "time": "Peak time [ns]",
                "std": "Amplitude Std. Dev. [ADC value]",
                "ani": "Amplitude [ADC value]",
                "img": "Amplitude [ADC value]",
                "charge": "Charge [ADC*ns]"
                         }


fpms_t = ['7-21', '7-22', '7-23', '7-24', '7-15', '7-16', '7-17', '7-18', '7-19', '7-10', '7-11', '7-12', '7-13', '7-14', '7-5', '7-6', '7-7', '7-8', '7-9', '7-0', '7-1', '7-20', '7-2', '7-3', '7-4']  # Hardcoded. Can probably be automated by reading the active slots in the reader object?
displaay = False

######## DO NOT CHANGE THIS MAPPING

ups_mods = { # Note this list modules that do not exist. This is because I just care about identifying positions which would flip upsidown if they were to be placed. 
    "0": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "2": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "3": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "5": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "6": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "8": [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
    "1": [1, 3, 6, 8, 11, 13, 16, 18, 21, 23],
    "4": [1, 3, 6, 8, 11, 13, 16, 18, 21, 23],
    "7": [1, 3, 6, 8, 11, 13, 16, 18, 21, 23],
}

fpms_exist = {
    "0": [8, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
    "2": [5, 6, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23],
    "3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "6": [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 18, 19],
    "8": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 15, 16],
    "1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "7": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
}

def load_csv(path):
    reader = csv.DictReader(open(path))

    result = {}
    for row in reader:
        for column, value in row.items():  # consider .iteritems() for Python 2
            result.setdefault(column, []).append(value)
    return result



def CTC_lab_im_map():
    """
    Calculates the grid index, which is used to go from (64) pixel data to (8, 8) lab image. Includes TARGET to SiPM mapping.

    :return: grid index
    :rtype: numba.typed.List
    """
    # #New
    # ch_nums = np.array([[21,20,17,16, 5, 4, 1, 0],
    #                     [23,22,19,18, 7, 6, 3, 2],
    #                     [29,28,25,24,13,12, 9, 8],
    #                     [31,30,27,26,15,14,11,10],
    #                     [53,52,49,48,37,36,33,32],
    #                     [55,54,51,50,39,38,35,34],
    #                     [61,60,57,56,45,44,41,40],
    #                     [63,62,59,58,47,46,43,42]])
    #Old
    ch_nums = np.array([[23,22,19,18, 4, 5, 0, 1],
                        [21,20,17,16, 6, 7, 2, 3],
                        [30,31,27,26,12,13, 8, 9],
                        [29,28,25,24,14,15,10,11],
                        [55,54,51,50,36,37,32,33],
                        [53,52,49,48,38,39,34,35],
                        [62,63,59,58,44,45,40,41],
                        [61,60,57,56,46,47,42,43]])

    
    ch_nums_1D = ch_nums.reshape(-1)
    ch_to_pos = dict(zip(ch_nums_1D, np.arange(64)))

    total_cells = 64
    indices = np.arange(total_cells).reshape(-1, int(np.sqrt(total_cells)))
    grid_ind = list()

    i, j = 0, 0
    ch_map = dict()
    ch_map = ch_to_pos
    pix_ind = np.array(indices[(8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
    for asic in range(4):
        for ch in range(16):
            grid_ind.append(int(pix_ind[ch_map[asic * 16 + ch]]))
            
    return grid_ind

def smart_mapping():
    refarr = np.array([15,11,14,10,13,9,12,8,7,3,6,2,5,1,4,0,19,23,18,22,17,21,16,20,27,30,26,31,25,29,24,28])
    refarr2 = refarr + 32
    refarr = list(refarr)
    refarr2 = list(refarr2)
    refarr = refarr + refarr2
    refarr = np.array(refarr)
    # return np.arange(64)
    return refarr

def fee_mapping():
    arr = smart_mapping()
    arrf = []
    for i in range(64):
        arrf.append(np.where(arr == i)[0][0])
    return arrf

grid_ind0 = CTC_lab_im_map()

smart_mapping0 = smart_mapping()
fee_mapping0 = fee_mapping()

def peak(arr):
    return np.max(arr)

def charge(arr, window = charge_window):
    peak_pos = np.argmax(arr)
    charge = np.sum(arr[max(0,peak_pos-int(window/2)):min(len(arr)+1,peak_pos+int(window/2)+1)])
    if charge_baseline_subtraction:
        charge = charge - (min(len(arr)+1,peak_pos+int(window/2)+1) - max(0,peak_pos-int(window/2)))*np.median(arr)
    
    return charge

def make_grid_module(vals, a = None, x0 = 0, y0 = 0, upsideup = True, grid_ind = grid_ind0, smart = smart_mapping0):
# grid_ind = CTC_lab_im_map()
    try:
        bb = a[0][0]
    except:
        a = np.empty(shape = (8+x0, 8+y0))
        if make_nans: a[:] = np.nan
        else: a[:] = 0.0
    bb = a.shape
    if bb[0] < x0 + 8:
        b = a
        a = np.empty(shape = (x0+8, a.shape[1]))
        a[:b.shape[0]][:b.shape[1]] = b
        bb = a.shape
    if bb[1] < y0 + 8:
        b = a
        a = np.empty(shape = (a.shape[0], y0+8))
        a[:b.shape[0]][:b.shape[1]] = b

    # vals = np.array(vals)
    # if len(vals) < 64:
    #     vals = np.pad(vals.astype(float), (0, 64 - len(vals)), constant_values=np.nan)
    if all(x == 0.0 for x in vals) and make_nans:
        vals[:] = np.nan
    for i, val in enumerate(vals):
        xy = grid_ind[i]
        y, x = xy // 8, xy % 8
        # x, y = xy % 8, xy // 8
        if upsideup:
            y =  7 - y
            x = 7 - x
        a[x0+x, y0+y] = val
    return np.array(a)

def make_grid_sector(valss, fpms,  a = None, x0 = 0, y0 = 0, sector = None, grid_ind = grid_ind0, smart=smart_mapping0):
    try:
        bb = a[0][0]
    except:
        a = np.empty(shape = (40+x0, 40+y0))
        a[:] = np.nan
    bb = a.shape
    if bb[0] < x0 + 40:
        b = a
        a = np.empty(shape = (x0+40, a.shape[1]))
        a[:b.shape[0]][:b.shape[1]] = b
    bb = a.shape
    if bb[1] < y0 + 40:
        b = a
        a = np.empty(shape = (a.shape[0], y0+40))
        a[:b.shape[0]][:b.shape[1]] = b
    valss = [vals for vals in valss]
    if not make_nans:
        if sector == None and len(fpms) > 0:
            sector, _ = fpms[0].split("-")
        elif sector == None:
            sector = "4" # this will be a blank picture I think
        for fp in fpms_exist[sector]:
            if f"{sector}-{fp}" not in fpms:
                fpms.append(f"{sector}-{fp}")
                valss.append([0.0 for _ in range(pxs_p_quad*quads_p_module)])

    for i, vals in zip(fpms, valss):
        sector, i = i.split("-")
        i = int(i)
        x00 = (i*8 % 40)
        y00 = (8*(4 - (i*8 // 40)))
        if i in ups_mods[str(sector)]:
            upsideup = True
        else:
            upsideup = False
        a = make_grid_module(vals, a, x0+x00, y0+y00, upsideup, grid_ind = grid_ind, smart = smart)
    return np.array(a)

def make_grid_camera(valss, fpms,  a = None, grid_ind = grid_ind0, smart=smart_mapping0):
    a = np.empty(shape = (120, 120))
    a[:] = np.nan

    fpmss = {"0": [],
             "1": [],
             "2": [],
             "3": [],
             "4": [],
             "5": [],
             "6": [],
             "7": [],
             "8": [],
    }

    valsss = {"0": [],
             "1": [],
             "2": [],
             "3": [],
             "4": [],
             "5": [],
             "6": [],
             "7": [],
             "8": [],
    }

    for val, fpm in zip(valss, fpms):
        sector, _ = fpm.split("-")
        fpmss[sector].append(fpm)
        valsss[sector].append(val)

    for i in range(9):
        sector = i
        i = f"{i}"
        fpms = fpmss[i]
        valss = valsss[i]
        # sector, _ = fpms[0].split("-")
        # sector = int(sector)
        x0 = (sector*40) % 120
        y0 = 80 - 40*(sector // 3)
        a = make_grid_sector(valss, fpms, a, x0, y0, grid_ind = grid_ind, smart = smart, sector = i)
    return np.array(a)

def camera_image(a, size = image_size, pixel_outline = pix_gridd, module_outline = mod_gridd, sector_outline = sector_gridd, vmin = low_bound, vmax = high_bound, titles = ["Event", "ADC Value"], save = None, show = displaay, colormap = colorscheme_default):

    fig, ax = plt.subplots(figsize=(size*1.2, size))

    im = ax.imshow(np.transpose(np.array(a)), cmap=colormap, interpolation="nearest", vmin = vmin, vmax = vmax)

    if pixel_outline:
        # Draw each pixel 
        for i in range(0, a.shape[1], 1):
            for j in range(0, a.shape[1], 1):
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=0.2*(size/8), edgecolor="lightgray", facecolor="none"
                ))

    if module_outline:
        # Draw 8x8 block outlines (light gray)
        for i in range(0, a.shape[1], 8):
            for j in range(0, a.shape[1], 8):
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 8, 8,
                    linewidth=0.8*(size/8), edgecolor="lightgray", facecolor="none"
                ))

    if sector_outline:
        # Draw 40x40 block outlines (light black)
        for i in range(0, a.shape[1], 40):
            for j in range(0, a.shape[1], 40):
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 40, 40,
                    linewidth=1.2*(size/8), edgecolor="#555555", facecolor="none"
                ))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[0], fontdict={'fontsize': title_font})
    # ax.set_title(titles[0])
    ax.set_aspect('equal')
    # ax.
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    cbar = plt.colorbar(im, ax=ax,)
    cbar.set_label(label=titles[1], fontsize = colorbar_legend_font)
    # cbar.set_label(label=titles[1])
    cbar.ax.tick_params(labelsize=colorbar_ticks_font, length=colorbar_ticks_font/3.5, width=colorbar_ticks_font/9)
    # plt.tight_layout()
    if show:
        plt.show()
    if save != None:
        fig.savefig(save)
    # plt.close()
    return fig, ax

def animate_frames(arrays, plot_fn, other_args = img_argss["ani"] + [["", colorbar_desc["ani"]], None, False], interval=250, save = None):
    """
    Parameters
    ----------
    arrays   : list of 2D numpy arrays, one per frame
    plot_fn  : your function that takes a 2D array and returns (fig, ax)
    interval : milliseconds between frames
    
    Returns
    -------
    fig, anim
    """
    interval = int(interval)
    other_args[7] = None
    other_args[8] = False
    fig, ax = plot_fn(arrays[0], *other_args)
    time_text = ax.text(
        0.45, -0.05,           # x, y in axes coordinates (top-left area)
        '',                    # starts empty
        transform=ax.transAxes,
        fontsize=time_legend_font,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.0)
    )
    im = ax.images[0]  # grab the AxesImage created by imshow

    def update(frame_idx):
        im.set_data(np.transpose(arrays[frame_idx]))
        current_time = frame_idx # in ns
        time_text.set_text(f't = {current_time} ns')
        
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(arrays),
        interval=interval,
        blit=True,
    )

    if save != None:
        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=int(1000/interval),
                                        metadata=dict(artist='Oris Salviano Neto'),
                                        bitrate=1800)
        anim.save(save, writer=writer)
    return fig, anim

def prep_lists(wfs_r0):
    events = []
    for wf in wfs_r0:
        wf = np.transpose(wf)
        wf = wf.astype(float)
        wf = wf.reshape(wf.shape[0], wf.shape[1] // 64, 64)

        events.append(wf)
    return events

def get_reader_object(filename):
    checks = glob.glob(filename)
    if len(checks) > 0:
        reader_r0 = target_io.WaveformArrayReader(filename)
    else:
        raise Exception(f"File '{filename}' does not exist.")
    return reader_r0

def get_event_from_reader(event = None, reader = None):
    if not (event == None or reader == None):
        if not reader.fR1:
            wfs_r0 = np.zeros((reader.fNPixels, reader.fNSamples), dtype=np.ushort)
            reader.GetR0Event(event, wfs_r0)
        else:
            wfs_r0 = np.zeros((reader.fNPixels, reader.fNSamples), dtype=np.float32)
            reader.GetR1Event(event, wfs_r0)
        wfs = prep_lists([wfs_r0])

        return wfs[0]
    else:
        print(event, reader)
        raise Exception("Either event or reader missing")


def display_event(path_to_file = None, 
                  im_type = None, 
                  disp_type = None, 
                  fpms = fpms_t, 
                  title = "",
                  font = None,
                  event = 0, 
                  frame = 0, 
                  save_im = None, 
                  display = True,  
                  reader = None,
                  wfs = None, 
                  ns_per_s = 6, 
                  img_args = None,
                  colorscheme = None,
                  colormap = None,
                  large_file = False,
                  reader_r0 = None):
    load_config_objects()
    if reader != None:
        reader_r0 = reader
    
    if im_type == None:
        im_type = default_image
    if disp_type == None:
        disp_type = default_display
    if colormap == None:
        colormap = colorscheme_default
    if colorscheme != None:
        colormap = colorscheme
    if disp_type == "minicam": disp_type = "sector"
    
    if font != None:
        global title_font
        global colorbar_legend_font
        global colorbar_ticks_font
        global time_legend_font
        title_font = font
        colorbar_legend_font = font
        colorbar_ticks_font = font
        time_legend_font = font


    if im_type not in modes_of_images:
        raise Exception(f"ERROR: wrong image type '{im_type}', choose among: 'peak', 'time', 'ani', 'img', 'charge'")

    if save_im != None:
        if im_type in ["img", "time", "peak", "charge", "std"]:

            if save_im[-4:][0] == ".":
                if save_im[-4:] not in [".png", ".jpg"]:
                    save_im = save_im[:-4] + ".png" 
            elif save_im[-5:][0] == ".":
                if save_im[-5:] not in [".jpeg"]:
                    save_im = save_im[:-5] + ".png"
            else:
                save_im = save_im + ".png"
        elif im_type in ["ani"]:

            if save_im[-4:][0] == ".":
                if save_im[-4:] not in [".gif", ".mp4"]:
                    save_im = save_im[:-4] + ".gif" 
            else:
                save_im = save_im + ".gif"


    img_args0 = img_argss[im_type].copy()

    img_args0.append([title, colorbar_desc[im_type]])
    if img_args == None:
        img_args = img_args0
    else:
        img_args = img_args[:6] + img_args0[-(len(img_args0)-min(len(img_args), 6)):]
        
    img_args.append(save_im)
    img_args.append(displaay)
    img_args.append(colormap)

    # arr_gen = make_grid_sector
    if disp_type == "camera":
        arr_gen = make_grid_camera
    elif disp_type == "sector":
        arr_gen = make_grid_sector
    else:
        print(f"Warning: display type '{disp_type}' not recognized. Using 'camera'")
        arr_gen = make_grid_camera
    
    
    if wfs == None:
        if reader_r0 == None:
            checks = glob.glob(path_to_file)
            if len(checks) > 0:
                reader_r0 = target_io.WaveformArrayReader(path_to_file)
            else:
                raise Exception(f"File '{path_to_file}' does not exist.")
        nss_p_ev = reader_r0.fNSamples
        nev_fil = reader_r0.fNEvents
        event = event % nev_fil
        frame = frame % nss_p_ev #ns
        if not reader_r0.fR1:
            if not large_file:
                wfs_r0 = [np.zeros((reader_r0.fNPixels, reader_r0.fNSamples), dtype=np.ushort) for _ in range(reader_r0.fNEvents)]
                for i in range(reader_r0.fNEvents):
                    reader_r0.GetR0Event(i, wfs_r0[i])
            else:
                wfs_r0_ev = np.zeros((reader_r0.fNPixels, reader_r0.fNSamples), dtype=np.ushort)
                reader_r0.GetR0Event(event, wfs_r0_ev)
                wfs_r0 = [wfs_r0_ev]
        else:
            if not large_file:
                wfs_r0 = [np.zeros((reader_r0.fNPixels, reader_r0.fNSamples), dtype=np.float32) for _ in range(reader_r0.fNEvents)]
                for i in range(reader_r0.fNEvents):
                    reader_r0.GetR1Event(i, wfs_r0[i])
            else:
                wfs_r0_ev = np.zeros((reader_r0.fNPixels, reader_r0.fNSamples), dtype=np.float32)
                reader_r0.GetR1Event(event, wfs_r0_ev)
                wfs_r0 = [wfs_r0_ev]

        wfs = prep_lists(wfs_r0)
        
        del reader_r0
    else:
        shapp = wfs[0].shape
        nss_p_ev = shapp[0]
        nev_fil = len(wfs)
        event = event % nev_fil
        frame = frame % nss_p_ev #ns
        
   
    

    if len(wfs) > 1:
        
        wf = wfs[event]
    else:
        wf = wfs[0]
        

    finn = [None, None]

    if save_im != None or display == True:

        if im_type == "img":

            a = arr_gen(wf[frame], fpms)
            if img_args[4] == None:
                img_args[4] = np.nanmin(a[a>0.0])
            if img_args[5] == None:
                img_args[5] = np.nanmax(a[a>0.0])

            finn = camera_image(a, *img_args) # {event}, frame {frame}

        elif im_type == "ani":

            a = np.array([arr_gen(wf[l], fpms) for l in range(nss_p_ev)])
            if img_args[4] == None:
                img_args[4] = np.nanmin(a[a>0.0])
            if img_args[5] == None:
                img_args[5] = np.nanmax(a[a>0.0])
            
            finn = animate_frames(a, camera_image, other_args=img_args, interval=1000/ns_per_s, save = save_im)
            if not display:
                plt.close()
                finn = [None, None]
        
        elif im_type == "time":

            wfs0a = np.moveaxis(np.array(wf), 0, 2) # wfs[event][time][fpm][fee_ch] --> wfs[event][fpm][fee_ch][time] = voltage
            # ns_per_ev = wfs0a.shape[2]
            wfs_fin = np.array([[[np.asarray(np.argmax(wfs0a[l][i]), dtype=float)] for i in range(wfs0a.shape[1])] for l in range(wfs0a.shape[0])])

            wfs_fin = np.moveaxis(wfs_fin,2,0) # wfs[event][fpm][fee_ch][time]
            a = arr_gen(wfs_fin[0], fpms)
            if img_args[4] == None:
                img_args[4] = np.nanmin(a[a>0.0])
            if img_args[5] == None:
                img_args[5] = np.nanmax(a[a>0.0])
            finn = camera_image(a, *img_args) # {event}, frame {frame}
        
        elif im_type == "peak":

            wfs0a = np.moveaxis(np.array(wf), 0, 2) # wfs[event][time][fpm][fee_ch] --> wfs[event][fpm][fee_ch][time] = voltage
            # ns_per_ev = wfs0a.shape[2]
            wfs_fin = np.array([[[peak(wfs0a[l][i])] for i in range(wfs0a.shape[1])] for l in range(wfs0a.shape[0])])

            wfs_fin = np.moveaxis(wfs_fin,2,0) # wfs[event][fpm][fee_ch][time]
            a = arr_gen(wfs_fin[0], fpms)
            
            if img_args[4] == None:
                img_args[4] = np.nanmin(a[a>0.0])
            if img_args[5] == None:
                img_args[5] = np.nanmax(a[a>0.0])
            finn = camera_image(a, *img_args) # {event}, frame {frame}
        
        elif im_type == "std":

            wfs0a = np.moveaxis(np.array(wf), 0, 2) # wfs[event][time][fpm][fee_ch] --> wfs[event][fpm][fee_ch][time] = voltage
            # ns_per_ev = wfs0a.shape[2]
            wfs_fin = np.array([[[np.nanstd(wfs0a[l][i])] for i in range(wfs0a.shape[1])] for l in range(wfs0a.shape[0])])

            wfs_fin = np.moveaxis(wfs_fin,2,0) # wfs[event][fpm][fee_ch][time]
            a = arr_gen(wfs_fin[0], fpms)
            
            if img_args[4] == None:
                img_args[4] = np.nanmin(a[a>0.0])
            if img_args[5] == None:
                img_args[5] = np.nanmax(a[a>0.0])
            finn = camera_image(a, *img_args) # {event}, frame {frame}

        elif im_type == "charge":

            wfs0a = np.moveaxis(np.array(wf), 0, 2) # wfs[event][time][fpm][fee_ch] --> wfs[event][fpm][fee_ch][time] = voltage
            # ns_per_ev = wfs0a.shape[2]
            wfs_fin = np.array([[[charge(wfs0a[l][i], window = charge_window)] for i in range(wfs0a.shape[1])] for l in range(wfs0a.shape[0])])

            wfs_fin = np.moveaxis(wfs_fin,2,0) # wfs[event][fpm][fee_ch][time]
            a = arr_gen(wfs_fin[0], fpms)
            
            if img_args[4] == None:
                img_args[4] = np.nanmin(a[a>0.0])
            if img_args[5] == None:
                img_args[5] = np.nanmax(a[a>0.0])
            finn = camera_image(a, *img_args) # {event}, frame {frame}
    
    return finn[0], finn[1], wfs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "--file", help="r0_file", default = None)
    parser.add_argument("-r", "--run", help="run to see. Overridden by a -file input", default=None)
    parser.add_argument("-sr", "--subrun", help="subrun to see. Overidden by a -file input", default="0")
    parser.add_argument("-ev", "--event", help="Event to see", default=0)
    parser.add_argument("-type", "--type", help="img or ani, for image or animation", default=default_image)
    parser.add_argument("-title", "--title", help="title of image", default="Run {0} Event {2}")
    parser.add_argument("-f", "--frame", help="Frame of event, if image", default=0)
    parser.add_argument("-dt", "--disp_type", help="Type of data run: camera, sector", default=default_display)
    parser.add_argument("-display", "--disp", help="whether to show image or animation", action="store_true")
    parser.add_argument("-save", "--save", help="path to save", default = None) 

    args = parser.parse_args()


    if args.file != None or args.run != None:
        
        im_type = args.type
        event = int(args.event)
        frame = int(args.frame)
        save_im = args.save
        displaay = args.disp
        disp_type = args.disp_type

        if args.file != None:
            path_to_file = args.file
            path_to_file = glob.glob(path_to_file)
            if len(path_to_file) == 0:
                raise Exception(f"File '{args.file}' not found.")
            else:
                path_to_file = path_to_file[-1]
            title = "Event {0}".format(event)
        else:
            path_to_file = path_to_file_def.format(args.run, args.subrun, "*")
            path_to_file = glob.glob(path_to_file)
            if len(path_to_file) == 0:
                raise Exception(f"No R0 or R1 file of the form '{path_to_file_def.format(args.run, args.subrun, '*')}' found.")
            else:
                path_to_file = path_to_file[-1]
                if len(path_to_file) > 1:
                    print(f"Warning: more than one file found (R0 and R1). Plotting '{path_to_file}'")

            title = args.title.format(args.run, args.subrun, event)

        if im_type not in modes_of_images:
            print(f"Unrecognized '{im_type}' image type. Using '{default_image}' display type")
            im_type = default_image
        
        img_args_f = img_argss[im_type]
        img_args_f.append([title, colorbar_desc[im_type]])
        display_event(path_to_file= path_to_file, 
                      im_type= im_type, 
                      disp_type= disp_type, 
                      fpms= fpms_t, 
                      event=event, 
                      frame= frame, 
                      save_im= save_im, 
                      display= displaay, 
                      img_args = img_args_f, 
                      title = title,
                      large_file=True)
        
    else:
        print("Provide path to r0 or r1 file with -file or run and subrun with -r and -sr")

if __name__ == "__main__":
    main()