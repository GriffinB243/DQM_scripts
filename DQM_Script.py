import target_io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import matplotlib.colors as colors
import matplotlib.patches as patches
from numba import njit
import time

#all options and changes listed here:
r0_file_location='/data/user/fbivens5020/mock_data/'#'folder where all the r0 files are'
pedestal_path='/data/wipac/CTA/targetcdata/run400032_pedestal.tcal'#'the chosen pedestal'
new_r1_file_location='/data/user/fbivens5020/mock_data/'#folder where live monitoring r1 files go'
old_r1_file_location=None #theoretically if you want it to use existing r1 files, option is currently broken
physical_metrics_location='/data/user/fbivens5020/mock_data/'#"folder where temps currents etc are stored, assuming they're all together"
modules=22 #the number of operable modules on the camera
type_number=10 #relates to how how sorting works, don't touch this

monitoring=True #run the loop
live=True #is data being taken live, if true will detect the most recent run and start looking at the next run after that
run_base=400214 #base for function that finds most recent run. only needs to be specific if there aren't any earlier runs in the file
initial_subrun=[400215,0] #first run and subrun to look at for existing data
final_subrun=[400215,5] #last subrun, can be used as a stopping point for live monitoring and looking at existing data

histograms_1d=True #true/false, will 1d histograms be generated at all
histograms_2d=True #true/false, will 2d histograms be generated at all
subrun_plots=True #true/false decides if plots will be generated and saved for all subruns as well (noncumulative)
boxes=True #true/false, will the cut boxes be visible on the histograms
noise_shower_regions=True #true/false, plots for the noise/shower region will be generated
flasher_regions=True #true/false, plots for flasher regions will be generated
tight_windows=False #true/false if false the region graphs look at the whole sorting box, if true they look at a tighter region for more detail between showers and flashers

extra_lines=False #true/false the lines for showers according to charge std and time std separately will be shown
resolution=1 #modifier on histogram bin sizes, 1 is a bin per second, 5 is a bin per fifth of a second, etc. it's set up to leave rate invariant
time_step=60E9 #modifier for the time scale, 60 billion is to take ns to min

subrun_plots=True #true/false decides if versions of
display_plots_path="/data/user/fbivens5020/DQM_scripts/DQM_plots/display_plots/"# path to folder of display files, these are the ones being overwritten through the loop
plots_save_path= "/data/user/fbivens5020/DQM_scripts/DQM_plots/subrun_plots/" #place to save all generated plot files for each run and subrun


#get the reader object with your r0_file_path, chosen pedestal path, and r1_file_path if one exists

def get_reader(r0_path, tcal_path, r1_path): #being actively used

    tcal_ped_path = tcal_path

    r0_file_path = r0_path

    file_path = r1_path
    
    os.system(f"apply_calibration_SCT -p {tcal_ped_path} -i {r0_file_path} -o {file_path}") # This will calibrate the data and make a r1 file
        
    reader = target_io.WaveformArrayReader(file_path)
    return reader

#start of stat collection sequence, in active use

def read_wfs(calfile, save=False, reader = None):
    if reader == None:
        reader = target_io.WaveformArrayReader(calfile, silent=True)
    all_wfs=[]
    times = []
    for ev in range(reader.fNEvents):
        wfs = np.zeros((reader.fNPixels, reader.fNSamples), dtype=np.float32)
        reader.GetR1Event(ev, wfs)
        times.append(float(reader.fTACK_time))
        all_wfs.append(wfs)
    all_wfs = np.array(all_wfs)
    if save:
        np.save(calfile.replace(".tio",".npy"), all_wfs)
    for wf in all_wfs:
        wf[4*64 + 14, :] = np.nan
        # for l in range(64):
        wf[5*64:6*64, :] = np.nan
        wf[13*64:14*64, :] = np.nan
        wf[21*64:, :] = np.nan
    return all_wfs, times

@njit
def get_max_time(all_wfs):
    t_maxs = np.zeros(all_wfs.shape[:-1]) # shape (n_events, n_channels)
    t_maxs[:, :] = np.argmax(all_wfs[:, :], axis=-1) # find the index of the maximum value along the time axis for each event and channel
    return t_maxs

@njit
def get_int_charges(all_wfs, int_win=2, charge_ceiling = None):
    n_events, n_channels, n_samples = all_wfs.shape
    int_charge = np.zeros((n_events, n_channels))
    for ev in range(n_events):
        for ch in range(n_channels):
            t_max = np.argmax(all_wfs[ev, ch])
            if t_max > n_samples - int_win - 1:
                t_max = n_samples - int_win - 1
            elif t_max < int_win:
                t_max = int_win
            int_charge[ev, ch] = all_wfs[ev, ch, t_max-int_win:t_max+int_win+1].sum()
        if charge_ceiling != None:
            for ch in range(n_channels):
                if int_charge[ev, ch] < charge_ceiling:
                    int_charge[ev, ch] = np.nan
    return int_charge


@njit
def get_event_stats(int_charges, peak_times):
    A = 30
    B = 15
    n_events, n_channels = int_charges.shape
    chg_means = np.zeros(n_events)
    chg_sums = np.zeros(n_events)
    chg_stds = np.zeros(n_events)
    time_means = np.zeros(n_events)
    time_stds = np.zeros(n_events)
    for n, chg_ev,  time_ev in zip(range(n_events), int_charges, peak_times):
        chg_means[n] = np.nanmean(chg_ev)
        chg_sums[n] = np.nansum(chg_ev)
        chg_stds[n] = np.nanstd(chg_ev)
        time_means[n] = np.nanmean(time_ev)
        time_stds[n] = np.nanstd(time_ev)

    return time_means, time_stds, chg_means, chg_stds

def collect_stats(reader): #being actively used

    all_wfs, timess = read_wfs(None, reader=reader)
    int_charges = get_int_charges(all_wfs, int_win=4)
    peak_times = get_max_time(all_wfs)
    stats_all = get_event_stats(int_charges, peak_times)


    return all_wfs, stats_all[0], stats_all[1], stats_all[2], stats_all[3], timess

#returns the sr_data object which is sr[0]: all wfs, sr[1]: mean time, sr[1][ev]: mean time for an event, sr[2]: time std
#sr[3]: charge mean, sr[4]: charge std, sr[5]: event time

def get_cuts(): #being actively used
    charge_mean_shower_max=2000
    charge_mean_shower_min=40

    charge_std_shower_max=2000
    charge_std_shower_min=40

    charge_mean_flasher_max=3750
    charge_mean_flasher_min=2000

    charge_std_flasher_max=1750
    charge_std_flasher_min=800

    time_std_shower_max=21
    time_std_shower_min=14

    time_std_flasher_max=18
    time_std_flasher_min=12
    
    return charge_mean_shower_min, charge_mean_shower_max, charge_std_shower_min, charge_std_shower_max, charge_mean_flasher_min, charge_mean_flasher_max, charge_std_flasher_min, charge_std_flasher_max, time_std_shower_min, time_std_shower_max, time_std_flasher_min, time_std_flasher_max

#establishes ranges for sorting boxes, make sure to have cuts=get_cuts
#sorting function, to be phased out but still works
def sort_data(sr_data, cuts, list=False): #Out
    ch_showers=[]
    t_showers=[]
    ch_flashers=[]
    t_flashers=[]
    ch_noise=[]
    t_noise=[]
    con_showers=[]
    con_flashers=[]
    con_noise=[]
    all_events=[]

    for ev in range(len(sr_data[0])):

        all_events.append(ev)

        if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[4][ev]>cuts[2] and sr_data[4][ev]<cuts[3]:
          ch_showers.append(ev)
        elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:# and sr_data[4][ev]>cuts[6] and sr_data[4][ev]<cuts[7]:
            ch_flashers.append(ev)
        else:
            ch_noise.append(ev)

        if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[2][ev]>cuts[8] and sr_data[2][ev]<cuts[9]:
            t_showers.append(ev)
        elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:# and sr_data[2][ev]>cuts[10] and sr_data[2][ev]<cuts[11]:
            t_flashers.append(ev)
        else: 
            t_noise.append(ev)
    for eve in ch_showers:
        if eve in t_showers:
            con_showers.append(eve)
    for eve in ch_flashers:
        if eve in t_flashers:
            con_flashers.append(eve)
    for eve in ch_noise:
        if eve in t_noise:
            con_noise.append(eve)


    all_events_data=np.zeros((6,len(all_events))) 
    for ind, ev in enumerate(all_events):
        all_events_data[0][ind]=ev
        all_events_data[1][ind]=sr_data[5][ev]

    charge_showers=np.zeros((6,len(ch_showers)))
    for ind, ev in enumerate(ch_showers):
        charge_showers[0][ind]=ev
        charge_showers[1][ind]=sr_data[5][ev]
    
    charge_flashers=np.zeros((6,len(ch_flashers)))
    for ind, ev in enumerate(ch_flashers):
        charge_flashers[0][ind]=ev
        charge_flashers[1][ind]=sr_data[5][ev]

    charge_noise=np.zeros((6,len(ch_noise)))
    for ind, ev in enumerate(ch_noise):
        charge_noise[0][ind]=ev
        charge_noise[1][ind]=sr_data[5][ev]

    time_showers=np.zeros((6,len(t_showers)))
    for ind, ev in enumerate(t_showers):
        time_showers[0][ind]=ev
        time_showers[1][ind]=sr_data[5][ev]

    time_flashers=np.zeros((6,len(t_flashers)))
    for ind, ev in enumerate(t_flashers):
        time_flashers[0][ind]=ev
        time_flashers[1][ind]=sr_data[5][ev]

    time_noise=np.zeros((6,len(t_noise)))
    for ind, ev in enumerate(t_noise):
        time_noise[0][ind]=ev
        time_noise[1][ind]=sr_data[5][ev]

    conf_showers=np.zeros((6,len(con_showers)))
    for ind, ev in enumerate(con_showers):
        conf_showers[0][ind]=ev
        conf_showers[1][ind]=sr_data[5][ev]

    conf_flashers=np.zeros((6,len(con_flashers)))
    for ind, ev in enumerate(con_flashers):
        conf_flashers[0][ind]=ev
        conf_flashers[1][ind]=sr_data[5][ev]

    conf_noise=np.zeros((6,len(con_noise)))
    for ind, ev in enumerate(con_noise):
        conf_noise[0][ind]=ev
        conf_noise[1][ind]=sr_data[5][ev]
        
    if list==True:
       print('Showers:',len(con_showers),'\nFlahsers:', len(con_flashers), '\nNoise:', len(con_noise), '\nCharge Showers:',len(ch_showers),'\nCharge Flashers:', len(ch_flashers),'\nCharge Noise:',len(ch_noise),"\nTime Showers:",len(t_showers),'\nTime Noise',len(t_noise))
       
    return all_events_data, conf_showers, conf_flashers, conf_noise, charge_showers, time_showers, charge_flashers, time_flashers, charge_noise, time_noise
    
#sorts the data into 9 lists, should be used to create the sorted_data object which has 9 sections with 2 indexes each
def real_new_sort(sr_data,subrun, sorted_run_data, cuts): #new and shiny and being used
   confirmations=[[],[],[],[],0]
   sorted_subrun=[[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]]]
   for ev in range(len(sr_data[0])):
        #all events data
        sorted_run_data[0][0].append(sr_data[5][ev]) #event time
        sorted_run_data[0][1].append(sr_data[3][ev]) #mean charge
        sorted_run_data[0][2].append(sr_data[4][ev]) #charge std
        sorted_run_data[0][3].append(sr_data[1][ev]) #mean time
        sorted_run_data[0][4].append(sr_data[2][ev]) #time std
        sorted_run_data[0][5].append(ev)#event id inside subrun
        sorted_run_data[0][6].append(subrun)# subrun id to make event id usable
        #again but for the subrun only
        sorted_subrun[0][0].append(sr_data[5][ev]) #event time
        sorted_subrun[0][1].append(sr_data[3][ev]) #mean charge
        sorted_subrun[0][2].append(sr_data[4][ev]) #charge std
        sorted_subrun[0][3].append(sr_data[1][ev]) #mean time
        sorted_subrun[0][4].append(sr_data[2][ev]) #time std
        sorted_subrun[0][5].append(ev)#event id inside subrun
        sorted_subrun[0][6].append(subrun)# subrun id to make event id usable

        if ev==0:
            sorted_run_data[0][7].append(0)
            sorted_subrun[0][7].append(0)
            subruns.append([subrun, sr_data[5][ev]]) #if first event delta t =0, also record the subrun start time
        else:
            sorted_run_data[0][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
            sorted_subrun[0][7].append(sr_data[5][ev]-sr_data[5][ev-1])

        if sr_data[3][ev]>cuts[1] and sr_data[3][ev]<cuts[0] and sr_data[4][ev]>cuts[3] and sr_data[4][ev]<cuts[2]:
            #charge showers
            confirmations[0].append(ev)
            sorted_run_data[4][0].append(sr_data[5][ev]) #event time
            sorted_run_data[4][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[4][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[4][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[4][4].append(sr_data[2][ev]) #time std
            sorted_run_data[4][5].append(ev)#event id inside subrun
            sorted_run_data[4][6].append(subrun)# subrun id to make event id usable
            #again but for the subrun only
            sorted_subrun[4][0].append(sr_data[5][ev]) #event time
            sorted_subrun[4][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[4][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[4][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[4][4].append(sr_data[2][ev]) #time std
            sorted_subrun[4][5].append(ev)#event id inside subrun
            sorted_subrun[4][6].append(subrun)# subrun id to make event id usable
            
            if ev==0:
                sorted_run_data[4][7].append(0)
                sorted_subrun[4][7].append(0)
                 #if first event delta t=0
            else:
                sorted_run_data[4][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[4][7].append(sr_data[5][ev]-sr_data[5][ev-1])
          
        elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:
            #actually just flashers now
            sorted_run_data[2][0].append(sr_data[5][ev]) #event time
            sorted_run_data[2][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[2][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[2][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[2][4].append(sr_data[2][ev]) #time std
            sorted_run_data[2][5].append(ev)#event id inside subrun
            sorted_run_data[2][6].append(subrun)# subrun id to make event id usable

            #again but for the subrun only
            sorted_subrun[2][0].append(sr_data[5][ev]) #event time
            sorted_subrun[2][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[2][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[2][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[2][4].append(sr_data[2][ev]) #time std
            sorted_subrun[2][5].append(ev)#event id inside subrun
            sorted_subrun[2][6].append(subrun)# subrun id to make event id usable
            if ev==0:
                sorted_run_data[2][7].append(0)
                sorted_subrun[2][7].append(0)
                 #if first event delta t=0
            else:
                sorted_run_data[2][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[2][7].append(sr_data[5][ev]-sr_data[5][ev-1])
        else:
            #charge noise i guess we're still doing this
            confirmations[1].append(ev)
            sorted_run_data[6][0].append(sr_data[5][ev]) #event time
            sorted_run_data[6][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[6][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[6][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[6][4].append(sr_data[2][ev]) #time std
            sorted_run_data[6][5].append(ev)#event id inside subrun
            sorted_run_data[6][6].append(subrun)# subrun id to make event id usable

            #again but for the subrun only
            sorted_subrun[6][0].append(sr_data[5][ev]) #event time
            sorted_subrun[6][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[6][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[6][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[6][4].append(sr_data[2][ev]) #time std
            sorted_subrun[6][5].append(ev)#event id inside subrun
            sorted_subrun[6][6].append(subrun)# subrun id to make event id usable
            if ev==0:
                sorted_run_data[6][7].append(0)
                sorted_subrun[6][7].append(0)
                 #if first event delta t=0
            else:
                sorted_run_data[6][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[6][7].append(sr_data[5][ev]-sr_data[5][ev-1])

        if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[2][ev]>cuts[8] and sr_data[2][ev]<cuts[9]:
            #time showers
            confirmations[2].append(ev)
            sorted_run_data[5][0].append(sr_data[5][ev]) #event time
            sorted_run_data[5][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[5][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[5][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[5][4].append(sr_data[2][ev]) #time std
            sorted_run_data[5][5].append(ev)#event id inside subrun
            sorted_run_data[5][6].append(subrun)# subrun id to make event id usable

            #again but for the subrun only
            sorted_subrun[5][0].append(sr_data[5][ev]) #event time
            sorted_subrun[5][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[5][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[5][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[5][4].append(sr_data[2][ev]) #time std
            sorted_subrun[5][5].append(ev)#event id inside subrun
            sorted_subrun[5][6].append(subrun)# subrun id to make event id usable
            if ev==0:
                sorted_run_data[5][7].append(0)
                sorted_subrun[5][7].append(0)
                 #if first event delta t=0
            else:
                sorted_run_data[5][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[5][7].append(sr_data[5][ev]-sr_data[5][ev-1])

        elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:
            #don't gotta do anything :)
            confirmations[4]+=1
        else: 
            #time noise
            confirmations[3].append(ev)
            sorted_run_data[7][0].append(sr_data[5][ev]) #event time
            sorted_run_data[7][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[7][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[7][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[7][4].append(sr_data[2][ev]) #time std
            sorted_run_data[7][5].append(ev)#event id inside subrun
            sorted_run_data[7][6].append(subrun)# subrun id to make event id usable

            #again but for the subrun only
            sorted_subrun[7][0].append(sr_data[5][ev]) #event time
            sorted_subrun[7][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[7][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[7][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[7][4].append(sr_data[2][ev]) #time std
            sorted_subrun[7][5].append(ev)#event id inside subrun
            sorted_subrun[7][6].append(subrun)# subrun id to make event id usable
            if ev==0:
                sorted_run_data[7][7].append(0)
                sorted_subrun[7][7].append(0)
                #if first event delta t=0
            else:
                sorted_run_data[7][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[7][7].append(sr_data[5][ev]-sr_data[5][ev-1])

        if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[4][ev]>cuts[2] and sr_data[4][ev]<cuts[3]and sr_data[2][ev]>cuts[8] and sr_data[2][ev]<cuts[9]:
            #confirmed showers
            sorted_run_data[1][0].append(sr_data[5][ev]) #event time
            sorted_run_data[1][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[1][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[1][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[1][4].append(sr_data[2][ev]) #time std
            sorted_run_data[1][5].append(ev)#event id inside subrun
            sorted_run_data[1][6].append(subrun)# subrun id to make event id usable

            #again but for the subrun only
            sorted_subrun[1][0].append(sr_data[5][ev]) #event time
            sorted_subrun[1][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[1][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[1][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[1][4].append(sr_data[2][ev]) #time std
            sorted_subrun[1][5].append(ev)#event id inside subrun
            sorted_subrun[1][6].append(subrun)# subrun id to make event id usable
            if ev==0:
                sorted_run_data[1][7].append(0)
                sorted_subrun[1][7].append(0)
                #if first event delta t=0
            else:
                sorted_run_data[1][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[1][7].append(sr_data[5][ev]-sr_data[5][ev-1])
        
        if ev in confirmations[1] and confirmations[3]:
            #confirmed noise
            sorted_run_data[3][0].append(sr_data[5][ev]) #event time
            sorted_run_data[3][1].append(sr_data[3][ev]) #mean charge
            sorted_run_data[3][2].append(sr_data[4][ev]) #charge std
            sorted_run_data[3][3].append(sr_data[1][ev]) #mean time
            sorted_run_data[3][4].append(sr_data[2][ev]) #time std
            sorted_run_data[3][5].append(ev)#event id inside subrun
            sorted_run_data[3][6].append(subrun)# subrun id to make event id usable

            #again but for the subrun only
            sorted_subrun[3][0].append(sr_data[5][ev]) #event time
            sorted_subrun[3][1].append(sr_data[3][ev]) #mean charge
            sorted_subrun[3][2].append(sr_data[4][ev]) #charge std
            sorted_subrun[3][3].append(sr_data[1][ev]) #mean time
            sorted_subrun[3][4].append(sr_data[2][ev]) #time std
            sorted_subrun[3][5].append(ev)#event id inside subrun
            sorted_subrun[3][6].append(subrun)# subrun id to make event id usable
            if ev==0:
                sorted_run_data[3][7].append(0)
                sorted_subrun[3][7].append(0)
                #if first event delta t=0
            else:
                sorted_run_data[3][7].append(sr_data[5][ev]-sr_data[5][ev-1]) #delta t
                sorted_subrun[3][7].append(sr_data[5][ev]-sr_data[5][ev-1])

   return sorted_run_data, sorted_subrun

#event rate histograms, being actively used
def event_rate_hists(current_sr, sorted_run_data, sorted_subrun, resolution, time_step, display_plots_path, plots_save_path, extra_lines=False, subrun_plots=False):
    modifier=resolution
    fig, ax = plt.subplots()

    ax.hist(sorted_run_data[0][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_data[0][0]))], bins = np.arange(sorted_run_data[0][0][0]/(time_step), sorted_run_data[0][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'All') 
    ax.hist(sorted_run_data[1][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_data[1][0]))], bins = np.arange(sorted_run_data[1][0][0]/(time_step), sorted_run_data[1][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Showers') 
    ax.hist(sorted_run_data[2][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_data[2][0]))], bins = np.arange(sorted_run_data[2][0][0]/(time_step), sorted_run_data[2][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Flashers') 
    ax.hist(sorted_run_data[3][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_data[3][0]))], bins = np.arange(sorted_run_data[3][0][0]/(time_step), sorted_run_data[3][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Other') 
        
    if extra_lines==True:
       ax.hist(sorted_run_data[4][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_data[4][0]))], bins = np.arange(sorted_run_data[4][0][0]/(time_step), sorted_run_data[4][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Charge Showers') 
       ax.hist(sorted_run_data[5][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_data[5][0]))], bins = np.arange(sorted_run_data[5][0][0]/(time_step), sorted_run_data[5][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Time Showers') 

    ax.legend(loc='upper left')
    ax.set_title(f"Event Rates Run {current_sr[0]}, Subruns 0-{current_sr[1]}")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Rate [Hz]")
    fig.savefig(f"{display_plots_path}event_rate_histogram.jpg")
    fig.savefig(f"{plots_save_path}run_{current_sr[0]}_event_rate_histogram.jpg")
    plt.close()
    
    if subrun_plots==True:

       fig, ax = plt.subplots()
       ax.hist(sorted_subrun[0][0]/(time_step), weights = [modifier for _ in range(len(sorted_subrun[0][0]))], bins = np.arange(sorted_subrun[0][0][0]/(time_step), sorted_subrun[0][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'All') 
       ax.hist(sorted_subrun[1][0]/(time_step), weights = [modifier for _ in range(len(sorted_subrun[1][0]))], bins = np.arange(sorted_subrun[1][0][0]/(time_step), sorted_subrun[1][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Showers') 
       ax.hist(sorted_subrun[2][0]/(time_step), weights = [modifier for _ in range(len(sorted_subrun[2][0]))], bins = np.arange(sorted_subrun[2][0][0]/(time_step), sorted_subrun[2][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Flashers') 
       ax.hist(sorted_subrun[3][0]/(time_step), weights = [modifier for _ in range(len(sorted_subrun[3][0]))], bins = np.arange(sorted_subrun[3][0][0]/(time_step), sorted_subrun[3][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Other')  
    
       if extra_lines==True:
          ax.hist(sorted_subrun[4][0]/(time_step), weights = [modifier for _ in range(len(sorted_subrun[4][0]))], bins = np.arange(sorted_subrun[4][0][0]/(time_step), sorted_subrun[4][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Charge Showers') 
          ax.hist(sorted_subrun[5][0]/(time_step), weights = [modifier for _ in range(len(sorted_subrun[5][0]))], bins = np.arange(sorted_subrun[5][0][0]/(time_step), sorted_subrun[5][0][-1]/(time_step), (1E9/time_step)/modifier), log=True, histtype = 'step', label = 'Time Showers')  
    
       ax.legend(loc='upper left')
       ax.set_title(f"Event Rates Run {current_sr[0]}, Subrun {current_sr[1]}")
       ax.set_xlabel("Time [min]")
       ax.set_ylabel("Rate [Hz]")
       fig.savefig(f"{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_event_rate.jpg")
       plt.close()

#2d histograms, being actively used
def sorting_hists_2d(cuts, current_sr, sorted_run_data, sr_data, display_plots_path, plots_save_path, subrun_plots=False, boxes=True, regions=True, flashers=True, tight=False):
   fig=plt.figure()
   ax=fig.add_subplot(111)
   ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

   if boxes==True:
       ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[2]), width=(cuts[1]-cuts[0]), height=(cuts[3]-cuts[2]), linewidth=1, color='green', fill=False))
       ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[6]), width=(cuts[5]-cuts[4]), height=(cuts[7]-cuts[6]), linewidth=1, color='red', fill=False))

   ax.set_title(f"Charge std vs Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)")
   ax.set_xlabel("Mean charge (ADC*ns)")
   ax.set_ylabel("Charge std (ADC*ns)")
   fig.savefig(f'{display_plots_path}charge_std_charge_mean_histogram.jpg')
   fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_histogram.jpg')
   plt.close()

   fig=plt.figure()
   ax=fig.add_subplot(111)
   ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

   if boxes==True:
       ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[8]), width=(cuts[1]-cuts[0]), height=(cuts[9]-cuts[8]), linewidth=1, color='green', fill=False))
       ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[10]), width=(cuts[5]-cuts[4]), height=(cuts[11]-cuts[10]), linewidth=1, color='red', fill=False))

   ax.set_title(f"Time std vs Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)")
   ax.set_xlabel("Mean charge (ADC*ns)")
   ax.set_ylabel("Time std (ns)")
   fig.savefig(f'{display_plots_path}time_std_charge_mean_histogram.jpg')
   fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_histogram.jpg')
   plt.close()

   if regions==True:
      
      if tight==True:
         charge_window=[[0,cuts[0]+100],[0,cuts[2]+100]]
         time_window=[[0, cuts[0]+100],[cuts[9]-3, cuts[9]+5]]
      else:
         charge_window=[[cuts[0]-50, cuts[1]+100],[cuts[2]-50, cuts[3]+100]]
         time_window=[[cuts[0]-50, cuts[1]+100],[cuts[8]-3, cuts[9]+3]]
    
      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=charge_window)

      if boxes==True:
         ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[2]), width=(cuts[1]-cuts[0]), height=(cuts[3]-cuts[2]), linewidth=1, color='green', fill=False))

      ax.set_title(f"Charge std vs Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)")
      ax.set_xlabel("Mean charge (ADC*ns)")
      ax.set_ylabel("Charge std (ADC*ns)")
      fig.savefig(f'{display_plots_path}charge_std_charge_mean_shower_region_histogram.jpg')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_shower_region_histogram.jpg')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=time_window)

      if boxes==True:
         ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[8]), width=(cuts[1]-cuts[0]), height=(cuts[9]-cuts[8]), linewidth=1, color='green', fill=False))

      ax.set_title(f"Time std vs Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)")
      ax.set_xlabel("Mean charge (ADC*ns)")
      ax.set_ylabel("Time std (ns)")
      fig.savefig(f'{display_plots_path}time_std_charge_mean_shower_region_histogram.jpg')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_shower_region_histogram.jpg')
      plt.close()
   
   if flashers==True:

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[cuts[4]-50,cuts[5]+50],[cuts[6]-50, cuts[7]+50]])

      if boxes==True:
         ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[6]), width=(cuts[5]-cuts[4]), height=(cuts[7]-cuts[6]), linewidth=1, color='red', fill=False))

      ax.set_title(f"Charge std vs Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)")
      ax.set_xlabel("Mean charge (ADC*ns)")
      ax.set_ylabel("Charge std (ADC*ns)")
      fig.savefig(f'{display_plots_path}charge_std_charge_mean_flasher_region_histogram.jpg')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_flasher_region_histogram.jpg')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=[[cuts[4]-50,cuts[5]+50],[cuts[10]-3,cuts[11]+3]])

      if boxes==True:
         ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[10]), width=(cuts[5]-cuts[4]), height=(cuts[11]-cuts[10]), linewidth=1, color='red', fill=False))

      ax.set_title(f"Time std vs Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)")
      ax.set_xlabel("Mean charge (ADC*ns)")
      ax.set_ylabel("Time std (ns)")
      fig.savefig(f'{display_plots_path}time_std_charge_mean_flasher_region_histogram.jpg')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_flasher_region_histogram.jpg')
      plt.close()

   if subrun_plots==True:
      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sr_data[1], sr_data[4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

      if boxes==True:
         ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[2]), width=(cuts[1]-cuts[0]), height=(cuts[3]-cuts[2]), linewidth=1, color='green', fill=False))
         ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[6]), width=(cuts[5]-cuts[4]), height=(cuts[7]-cuts[6]), linewidth=1, color='red', fill=False))

      ax.set_title(f"Charge std vs Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)")
      ax.set_xlabel("Mean charge (ADC*ns)")
      ax.set_ylabel("Charge std (ADC*ns)")
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_charge_mean_histogram.jpg')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sr_data[3], sr_data[2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

      if boxes==True:
         ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[8]), width=(cuts[1]-cuts[0]), height=(cuts[9]-cuts[8]), linewidth=1, color='green', fill=False))
         ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[10]), width=(cuts[5]-cuts[4]), height=(cuts[11]-cuts[10]), linewidth=1, color='red', fill=False))

      ax.set_title(f"Time std vs Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)")
      ax.set_xlabel("Mean charge (ADC*ns)")
      ax.set_ylabel("Time std (ns)")
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_histogram.jpg')
      plt.close()

      if regions==True:
      
         if tight==True:
            charge_window=[[0,cuts[0]+100],[0,cuts[2]+100]]
            time_window=[[0, cuts[0]+100],[cuts[9]-3, cuts[9]+5]]
         else:
            charge_window=[[cuts[0]-50, cuts[1]+100],[cuts[2]-50, cuts[3]+100]]
            time_window=[[cuts[0]-50, cuts[1]+100],[cuts[8]-3, cuts[9]+3]]
    
         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=charge_window)

         if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[2]), width=(cuts[1]-cuts[0]), height=(cuts[3]-cuts[2]), linewidth=1, color='green', fill=False))

         ax.set_title(f"Charge std vs Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)")
         ax.set_xlabel("Mean charge (ADC*ns)")
         ax.set_ylabel("Charge std (ADC*ns)")
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_shower_region_histogram.jpg')
         plt.close()

         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=time_window)

         if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[0],cuts[8]), width=(cuts[1]-cuts[0]), height=(cuts[9]-cuts[8]), linewidth=1, color='green', fill=False))

         ax.set_title(f"Time std vs Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)")
         ax.set_xlabel("Mean charge (ADC*ns)")
         ax.set_ylabel("Time std (ns)")
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_shower_region_histogram.jpg')
         plt.close()
         
      if flashers==True:

         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[cuts[4]-50,cuts[5]+50],[cuts[6]-50, cuts[7]+50]])

         if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[6]), width=(cuts[5]-cuts[4]), height=(cuts[7]-cuts[6]), linewidth=1, color='red', fill=False))

         ax.set_title(f"Charge std vs Mean Charge, Run {current_sr[0]}, Subruns{current_sr[1]} (Flasher Region)")
         ax.set_xlabel("Mean charge (ADC*ns)")
         ax.set_ylabel("Charge std (ADC*ns)")
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_charge_mean_flasher_region_histogram.jpg')
         plt.close()

         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=[[cuts[4]-50,cuts[5]+50],[cuts[10]-3,cuts[11]+3]])

         if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[4],cuts[10]), width=(cuts[5]-cuts[4]), height=(cuts[11]-cuts[10]), linewidth=1, color='red', fill=False))

         ax.set_title(f"Time std vs Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (Flasher Region)")
         ax.set_xlabel("Mean charge (ADC*ns)")
         ax.set_ylabel("Time std (ns)")
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_flasher_region_histogram.jpg')
         plt.close()

#1d histograms, being used currently but seems buggy
def sorting_hists_1d(cuts, current_sr, sorted_run_data, sr_data, display_plots_path, plots_save_path, subrun_plots=False, boxes=True, regions=True, flashers=True):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][2], bins = 200, log=True)

    if boxes==True:
        ax.add_patch(patches.Rectangle(xy=(cuts[0],0), width=(cuts[1]-cuts[0]), height=(1400), linewidth=1, color='green', fill=False))
        ax.add_patch(patches.Rectangle(xy=(cuts[4],0), width=(cuts[5]-cuts[4]), height=(1400), linewidth=1, color='red', fill=False))

    ax.set_title(f"Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)")
    ax.set_xlabel("Mean charge (ADC*ns)")
    ax.set_ylabel("Counts")
    fig.savefig(f'{display_plots_path}charge_mean_histogram.jpg')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_mean_histogram.jpg')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][3], bins = 200, log=True)

    if boxes==True:
        ax.add_patch(patches.Rectangle(xy=(cuts[2],0), width=(cuts[3]-cuts[2]), height=(1400), linewidth=1, color='green', fill=False))
        ax.add_patch(patches.Rectangle(xy=(cuts[6],0), width=(cuts[7]-cuts[6]), height=(1400), linewidth=1, color='red', fill=False))

    ax.set_title(f"Charge Std, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)")
    ax.set_xlabel("Charge Std (ADC*ns)")
    ax.set_ylabel("Counts")
    fig.savefig(f'{display_plots_path}charge_std_histogram.jpg')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_histogram.jpg')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][4], bins = 200, log=True)
    ax.set_title(f"Mean Peak Time, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)")
    ax.set_xlabel("Mean Peak Time (ns)")
    ax.set_ylabel("Counts")
    fig.savefig(f'{display_plots_path}time_mean_histogram.jpg')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_mean_histogram.jpg')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][5], bins = 200, log=True)

    if boxes==True:
        ax.add_patch(patches.Rectangle(xy=(cuts[8],0), width=(cuts[9]-cuts[8]), height=(1400), linewidth=1, color='green', fill=False))
        ax.add_patch(patches.Rectangle(xy=(cuts[10],0), width=(cuts[11]-cuts[10]), height=(1400), linewidth=1, color='red', fill=False))

    ax.set_title(f"Peak Time Std, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)")
    ax.set_xlabel("Peak Time Std (ns)")
    ax.set_ylabel("Counts")
    fig.savefig(f'{display_plots_path}time_std_histogram.jpg')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_histogram.jpg')
    plt.close()

    if regions==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][2], bins = 200, log=True, range=(cuts[0]-50, cuts[1]+50))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[0],0), width=(cuts[1]-cuts[0]), height=(1400), linewidth=1, color='green', fill=False))

        ax.set_title(f"Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)")
        ax.set_xlabel("Mean Charge (ADC*ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{display_plots_path}charge_mean_shower_region_histogram.jpg')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_mean_shower_region_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][3], bins = 200, log=True, range=(cuts[2]-50,cuts[3]+50))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[2],0), width=(cuts[3]-cuts[2]), height=(1400), linewidth=1, color='green', fill=False))

        ax.set_title(f"Charge Std, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)")
        ax.set_xlabel("Charge Std (ADC*ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{display_plots_path}charge_std_shower_region_histogram.jpg')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_shower_region_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][5], bins = 200, log=True, range=(cuts[8]-3, cuts[9]+3))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[8],0), width=(cuts[9]-cuts[8]), height=(1400), linewidth=1, color='green', fill=False))

        ax.set_title(f"Peak Time Std, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)")
        ax.set_xlabel("Peak Time Std (ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{display_plots_path}time_std_shower_region_histogram.jpg')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_shower_region_histogram.jpg')
        plt.close()

    if flashers==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][2], bins = 200, log=True, range=(cuts[4]-50, cuts[5]+50))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[4],0), width=(cuts[5]-cuts[4]), height=(1400), linewidth=1, color='red', fill=False))

        ax.set_title(f"Mean Charge, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)")
        ax.set_xlabel("Mean Charge (ADC*ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{display_plots_path}charge_mean_flasher_region_histogram.jpg')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_mean_flasher_region_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][3], bins = 200, log=True, range=(cuts[6]-50,cuts[7]+50))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[6],0), width=(cuts[7]-cuts[6]), height=(1400), linewidth=1, color='red', fill=False))

        ax.set_title(f"Charge Std, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)")
        ax.set_xlabel("Charge Std (ADC*ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{display_plots_path}charge_std_flasher_region_histogram.jpg')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_flasher_region_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][5], bins = 200, log=True, range=(cuts[10]-3, cuts[11]+3))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[10],0), width=(cuts[11]-cuts[10]), height=(1400), linewidth=1, color='red', fill=False))

        ax.set_title(f"Peak Time Std, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)")
        ax.set_xlabel("Peak Time Std (ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{display_plots_path}time_std_flasher_region_histogram.jpg')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_flasher_region_histogram.jpg')
        plt.close()

    if subrun_plots==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[3], bins = 200, log=True)

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[0],0), width=(cuts[1]-cuts[0]), height=(1400), linewidth=1, color='green', fill=False))
            ax.add_patch(patches.Rectangle(xy=(cuts[4],0), width=(cuts[5]-cuts[4]), height=(1400), linewidth=1, color='red', fill=False))

        ax.set_title(f"Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)")
        ax.set_xlabel("Mean charge (ADC*ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_mean_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[4], bins = 200, log=True)

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[2],0), width=(cuts[3]-cuts[2]), height=(1400), linewidth=1, color='green', fill=False))
            ax.add_patch(patches.Rectangle(xy=(cuts[6],0), width=(cuts[7]-cuts[6]), height=(1400), linewidth=1, color='red', fill=False))

        ax.set_title(f"Charge Std, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)")
        ax.set_xlabel("Charge Std (ADC*ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[1], bins = 200, log=True)
        ax.set_title(f"Mean Peak Time, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)")
        ax.set_xlabel("Mean Peak Time (ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_mean_histogram.jpg')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[2], bins = 200, log=True)

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[8],0), width=(cuts[9]-cuts[8]), height=(1400), linewidth=1, color='green', fill=False))
            ax.add_patch(patches.Rectangle(xy=(cuts[10],0), width=(cuts[11]-cuts[10]), height=(1400), linewidth=1, color='red', fill=False))

        ax.set_title(f"Peak Time Std, Run {current_sr[0]}, Subruns {current_sr[1]} (All Events)")
        ax.set_xlabel("Peak Time Std (ns)")
        ax.set_ylabel("Counts")
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_histogram.jpg')
        plt.close()

        if regions==True:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[3], bins = 200, log=True, range=(cuts[0]-50, cuts[1]+50))

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[0],0), width=(cuts[1]-cuts[0]), height=(1400), linewidth=1, color='green', fill=False))

            ax.set_title(f"Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)")
            ax.set_xlabel("Mean Charge (ADC*ns)")
            ax.set_ylabel("Counts")
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_mean_shower_region_histogram.jpg')
            plt.close()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[4], bins = 200, log=True, range=(cuts[2]-50,cuts[3]+50))

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[2],0), width=(cuts[3]-cuts[2]), height=(1400), linewidth=1, color='green', fill=False))

            ax.set_title(f"Charge Std, Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)")
            ax.set_xlabel("Charge Std (ADC*ns)")
            ax.set_ylabel("Counts")
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_shower_region_histogram.jpg')
            plt.close()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[2], bins = 200, log=True, range=(cuts[8]-3, cuts[9]+3))

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[8],0), width=(cuts[9]-cuts[8]), height=(1400), linewidth=1, color='green', fill=False))

            ax.set_title(f"Peak Time Std, Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)")
            ax.set_xlabel("Peak Time Std (ns)")
            ax.set_ylabel("Counts")
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_shower_region_histogram.jpg')
            plt.close()

        if flashers==True:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[3], bins = 200, log=True, range=(cuts[4]-50, cuts[5]+50))

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[4],0), width=(cuts[5]-cuts[4]), height=(1400), linewidth=1, color='red', fill=False))

            ax.set_title(f"Mean Charge, Run {current_sr[0]}, Subrun {current_sr[1]} (Flasher Region)")
            ax.set_xlabel("Mean Charge (ADC*ns)")
            ax.set_ylabel("Counts")
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_mean_flasher_region_histogram.jpg')
            plt.close()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[4], bins = 200, log=True, range=(cuts[6]-50,cuts[7]+50))

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[6],0), width=(cuts[7]-cuts[6]), height=(1400), linewidth=1, color='red', fill=False))

            ax.set_title(f"Charge Std, Run {current_sr[0]}, Subrun {current_sr[1]} (Flasher Region)")
            ax.set_xlabel("Charge Std (ADC*ns)")
            ax.set_ylabel("Counts")
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_flasher_region_histogram.jpg')
            plt.close()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[2], bins = 200, log=True, range=(cuts[10]-3, cuts[11]+3))

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[10],0), width=(cuts[11]-cuts[10]), height=(1400), linewidth=1, color='red', fill=False))

            ax.set_title(f"Peak Time Std, Run {current_sr[0]}, Subrun {current_sr[1]} (Flasher Region)")
            ax.set_xlabel("Peak Time Std (ns)")
            ax.set_ylabel("Counts")
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_flasher_region_histogram.jpg')
            plt.close()

# #older event rate function, likely to be deleted
# def event_rate(sorted_data, sr_data, run_id, sr_number, save_location, mod=1, test=False):
#     modifier=mod
#     # fig=plt.figure()
#     # ax=fig.add_subplot(111)
#     fig, ax = plt.subplots()
#     ax.hist(sr_data[5], weights = [modifier for _ in range(len(sr_data[5]))], bins = np.arange(sr_data[5][0], sr_data[5][-1], 1E9/modifier), log=True, histtype = 'step', label = 'All') 
#     ax.hist(sorted_data[0][1], weights = [modifier for _ in range(len(sorted_data[0][1]))], bins = np.arange(sorted_data[0][1][0], sorted_data[0][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'Showers') 
#     ax.hist(sorted_data[1][1], weights = [modifier for _ in range(len(sorted_data[1][1]))], bins = np.arange(sorted_data[1][1][0], sorted_data[1][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'Flashers') 
#     ax.hist(sorted_data[2][1], weights = [modifier for _ in range(len(sorted_data[2][1]))], bins = np.arange(sorted_data[2][1][0], sorted_data[2][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'Other') 
    
#     if test==True:
#        ax.hist(sorted_data[3][1], weights = [modifier for _ in range(len(sorted_data[3][1]))], bins = np.arange(sorted_data[3][1][0], sorted_data[3][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'charge showers') 
#        ax.hist(sorted_data[6][1], weights = [modifier for _ in range(len(sorted_data[6][1]))], bins = np.arange(sorted_data[6][1][0], sorted_data[6][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'time showers') 
   
#     ax.legend(loc='upper left')
#     ax.set_title(f"Event Rates Run {run_id}, subrun {sr_number}")
#     ax.set_xlabel("Time [ns]")
#     ax.set_ylabel("Rate [Hz]")
#     fig.savefig(save_location)
#     print("summary plot B has actually been generated")
#     plt.show()

#makes histogram of rates with or without some test lines that help with refining box cuts, to be deleted
# def event_rate_summary(run_id, sr_number, r0_location, tcal_location, r1_location, save_location, test=False, resolution=1):

#     sr_id=sr_number
#     r0_file=r0_location+'run'+str(run_id)+'_subrun'+str(sr_id)+'_r0.tio'

#     tcal_file=tcal_location

#     r1_file=r1_location+'run'+str(run_id)+'_subrun'+str(sr_id)+'.r1'

#     reader=get_reader(r0_file, tcal_file, r1_file)

#     sr_data=collect_stats(reader) 


#     cuts=get_cuts()
    
#     get_hists(sr_data, cuts, display=test, regions= test, flashers= test, boxes= test)

#     sorted_data=sort_data(sr_data, cuts, list=test)

#     event_rate(sorted_data, sr_data, run_id, sr_number, save_location, test=test, mod=resolution)

    #Produces the whole nice graph thing

#OTHER METRICS FUNCTION in use

def physical_summary(current_sr, run_data, physical_metrics_location, display_plots_path, plots_save_path, time_step, modules=22):
    
    fpmTemp_list=[]
    feeTemp_list=[]
    hv_list=[]
    current_list=[]

    for sr in range(current_sr[1]+1):

      fpmTemp_data=np.load(f'{physical_metrics_location}temperatures_FPMs_run{current_sr[0]}_subrun{sr}.npy')
      fpmTemp_list.append(fpmTemp_data)

      feeTemp_data=np.load(f'{physical_metrics_location}temperatures_FEEs_run{current_sr[0]}_subrun{sr}.npy')
      feeTemp_list.append(feeTemp_data)

      hv_data=np.load(f'{physical_metrics_location}HV_FPMs_run{current_sr[0]}_subrun{sr}.npy')
      hv_list.append(hv_data)

      current_data=np.load(f'{physical_metrics_location}Current_FEEs_run{current_sr[0]}_subrun{sr}.npy')
      current_list.append(current_data)

    fpmTemps=np.zeros((modules*4,4,len(fpmTemp_list[:])))
    feeTemps=np.zeros((modules*2,4,len(feeTemp_list[:])))
    hv=np.zeros((modules,4,len(hv_list[:])))
    current=np.zeros((modules,4,len(current_list[:])))

    for sr in range(current_sr[1]+1):

        for quad in range(modules*4):
           fpmTemps[quad][0][sr]=sr
           fpmTemps[quad][1][sr]=fpmTemp_list[sr][quad]
           fpmTemps[quad][2][sr]=quad//4
           fpmTemps[quad][3][sr]=run_data[sr][5][0]

        for board in range(modules*2):
           feeTemps[board][0][sr]=sr
           feeTemps[board][1][sr]=feeTemp_list[sr][board]
           feeTemps[board][2][sr]=board//2
           feeTemps[board][3][sr]=run_data[sr][5][0]

        for mod in range(modules):
           hv[mod][0][sr]=sr
           hv[mod][1][sr]=hv_list[sr][mod]
           hv[mod][2][sr]=mod
           hv[mod][3][sr]=run_data[sr][5][0]
           current[mod][0][sr]=sr
           current[mod][1][sr]=current_list[sr][mod]
           current[mod][2][sr]=mod
           current[mod][3][sr]=run_data[sr][5][0]

    fig, (ax1,ax2,ax3,ax4)=plt.subplots(4,1)

    for quad in range(modules*4):
       ax1.plot(fpmTemps[quad][3]/(time_step), fpmTemps[quad][1])
    ax1.set_ylabel("FPM Temp (C)")

    for board in range(modules*2):
       ax2.plot(feeTemps[board][3]/(time_step), feeTemps[board][1])
    ax2.set_ylabel("FEE Temp (C)")

    for mod in range(modules):
       ax3.plot(hv[mod][3]/(time_step),hv[mod][1])
       ax4.plot(current[mod][3]/(time_step),current[mod][1])
    ax3.set_ylabel("HV (V)")
    ax4.set_ylabel("Current (A)")
    ax4.set_xlabel("Time (min)")
    fig.tight_layout()
    #fig.set_label(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]}, Physical Metrics") #should be a title for the thing, not currently working
    fig.savefig(f'{display_plots_path}physical_metrics_plots.jpg')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_physical_metrics.jpg')
    plt.close()
    
#CONDENSED FUNCTION, to be deleted

# def sr_summary(run_id, sr_number, metrics_location, r0_location, tcal_location, r1_location, ev_save_location, phys_save_location, test=False, resolution=1, modules=22):

#     event_rate_summary(run_id, sr_number, r0_location, tcal_location, r1_location, ev_save_location, test=False, resolution=resolution)
#     print("Summary plot B has been generated")

#     physical_summary(run_id, sr_number, metrics_location, phys_save_location, modules=modules)
#     print("Summary plot A has been generated")

#     print(f"Run {run_id} sr{sr_number} summary")

#NEWEST FILE DETECTOR function, in use

def get_new_sr(physical_metrics_location, run_base=400196, subrun_base=0): #being actively used

   run=None
   while run==None:
      if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{run_base+1}_subrun0.npy")==True:
         run_base=run_base+1
      else:
         run=run_base

   subrun=None
   while subrun==None:
      if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{run}_subrun{subrun_base+1}.npy")==True:
         subrun_base=subrun_base+1
      else:
         subrun=subrun_base
   return run, subrun

# active=False

# while active==True: #This is the important cell that manages everything but it's rough and being phased out
   
#     current_sr=get_new_sr()
#     print(f"Latest Subrun: Run {current_sr[0]} sr{current_sr[1]}")

#     if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{current_sr[0]}_subrun{current_sr[1]}.npy")==True:
#        tim_n = time.time()
#        sr_summary(current_sr[0], current_sr[1], physical_metrics_location, r0_file_location, pedestal_path, new_r1_file_location, summary_plots_location_1, summary_plots_location_2)
#        print(f'Summary for Subrun {current_sr[1]} took {time.time()-tim_n}s')
#     else:
#         print("This file doesn't exist yet")
#         time.sleep(5)
#     next_sr=get_new_sr()
#     while current_sr==next_sr:
#        next_sr=get_new_sr()

#        if current_sr[1]==last_subrun: #This line just stops this from going forever, comment out when not working with simulator
#            active=False
#            break

total_run_data=[]
run_data_sorted=[]
live_monitoring= False
#live monitoring loop is not fully broken but the histograms will turn out very wonky if you try
while live_monitoring==True:

    current_sr=get_new_sr(physical_metrics_location, run_base=run_base, subrun_base=0)
    print(f'\nLatest Subrun: run {current_sr[0]}, subrun {current_sr[1]}')

    if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{current_sr[0]}_subrun{current_sr[1]}.npy")==True:

        tim_n=time.time()

        subrun_number=current_sr[1]+1

        r0_file=r0_file_location+'run'+str(current_sr[0])+'_subrun'+str(current_sr[1])+'_r0.tio'
        tcal_file=pedestal_path
        r1_file=new_r1_file_location+'run'+str(current_sr[0])+'_subrun'+str(current_sr[1])+'.r1'

        reader=get_reader(r0_file, tcal_file, r1_file) #get reader data
        time_s=time.time()
        sr_data=collect_stats(reader) #get useable stats from reader
        cuts=get_cuts() #get histogram cuts
        total_run_data.append(sr_data)#puts data for newest sr in the list for the whole run with shape [subrun][type of metric][event]
        sorted_subrun=sort_data(sr_data,cuts) #sort sr data for the subrun
        run_data_sorted.append(sorted_subrun) #put that sorted data in the runwide list of sorted data

        types_list=[[],[],[],[],[],[],[],[],[],[]] #list of types of events events can be sorted into, should have length type_number
        
        for sr in range(subrun_number):

            for type in range(type_number):
            
                for event in range(len(run_data_sorted[sr][type][0])):

                    types_list[type].append([run_data_sorted[sr][type][0][event],sr]) #loop that appends sorted events to appropriate lists

        all_events_data=np.zeros((6,len(types_list[0])))
        conf_shower_data=np.zeros((6,len(types_list[1])))
        conf_flasher_data=np.zeros((6,len(types_list[2])))
        conf_noise_data=np.zeros((6,len(types_list[3])))
        charge_shower_data=np.zeros((6,len(types_list[4])))
        time_shower_data=np.zeros((6,len(types_list[5])))
        charge_flasher_data=np.zeros((6,len(types_list[6])))
        time_flasher_data=np.zeros((6,len(types_list[7])))
        charge_noise_data=np.zeros((6,len(types_list[8])))
        time_noise_data=np.zeros((6,len(types_list[9]))) #arrays for the cumulative data of different types, also gives canon order for the event lists

        sorted_run_data=[all_events_data, conf_shower_data, conf_flasher_data, conf_noise_data, charge_shower_data, time_shower_data, charge_flasher_data, time_flasher_data, charge_noise_data, time_noise_data]

        for type in range(type_number): #loop that puts all the data in the right place and shape to work with the graphing functions
            for event in range(len(types_list[type])):
                sorted_run_data[type][0][event]=types_list[type][event][0]
                sorted_run_data[type][1][event]=total_run_data[int(types_list[type][event][1])][5][int(types_list[type][event][0])]
                sorted_run_data[type][2][event]=total_run_data[int(types_list[type][event][1])][3][int(types_list[type][event][0])]
                sorted_run_data[type][3][event]=total_run_data[int(types_list[type][event][1])][4][int(types_list[type][event][0])]
                sorted_run_data[type][4][event]=total_run_data[int(types_list[type][event][1])][1][int(types_list[type][event][0])]
                sorted_run_data[type][5][event]=total_run_data[int(types_list[type][event][1])][2][int(types_list[type][event][0])]

        print(f'\nsorting the data took {time.time()-time_s} s')
        #event rate histograms here
        time_h=time.time()
        event_rate_hists(current_sr, sorted_run_data, sorted_subrun, resolution, time_step, display_plots_path, plots_save_path, extra_lines=extra_lines, subrun_plots=subrun_plots)#event rate histograms for overall run and subrun
        print(f'\n event rate histograms took {time.time()-time_h} s\n')
        #2d histogram function here
        time_ht=time.time()
        sorting_hists_2d(cuts, current_sr, sorted_run_data, sr_data, display_plots_path, plots_save_path, subrun_plots=subrun_plots, boxes=boxes, regions=noise_shower_regions, flashers=flasher_regions, tight=tight_windows)
        print(f'\n 2d histograms took {time.time()-time_ht} s\n')
        #1d histogram function
        time_ho=time.time()
        sorting_hists_1d(cuts, current_sr, sorted_run_data, sr_data, display_plots_path, plots_save_path, subrun_plots=subrun_plots, boxes=boxes, regions=noise_shower_regions, flashers=flasher_regions)
        print(f'\n 1d histograms took {time.time()-time_ho} s\n')
        #physical metrics graph function
        time_p=time.time()
        physical_summary(current_sr, total_run_data, physical_metrics_location, display_plots_path, plots_save_path, time_step, modules=modules)
        print(f'\n physical metrics took {time.time()-time_p} s\n')
        #'heat' maps/camera visualizations function here

        #space to add other graphing options that take 
        print(f'\nsummary of run {current_sr[0]} subrun {current_sr[1]} took {time.time()-tim_n} s\n')
    else:
        print("this file doesn't exist yet")

    next_sr=get_new_sr(physical_metrics_location, run_base=run_base, subrun_base=0)

    while current_sr==next_sr: #loop that keeps us waiting for another subrun and stops us at run 400196 sr15
        next_sr=get_new_sr(physical_metrics_location, run_base=run_base, subrun_base=0)
        #this loop should work right and break when the run ticks over because the get_new_sr function looks for a newest run first
        #so it should find that get_new_sr()[0]=0 and that should break this loop on its own
        if [current_sr]==last_subrun:
            live_monitoring=False
            print(f"reached last subrun: {last_subrun}")
            break
        if next_sr[0]!=current_sr[0]:
            print(f'run {current_sr[0]} has ended')
            break #should detect when a run has ticked over to break the loop of waiting for a new subrun if the above loop doesn't work

current_target=initial_subrun
runs=[]
subruns=[]
sorted_run_data_format=[[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]]]
sorted_run_data=sorted_run_data_format
run_data=[]
while monitoring==True:
    print(f'\nlooking at run {current_target[0]}, sub-run {current_target[1]}')

    if os.path.exists(f'/data/user/fbivens5020/mock_data/run{current_target[0]}_subrun{current_target[1]}_r0.tio'):
        print(f'\nr0 file for {current_target} found')
        if os.path.exists(f'/data/user/fbivens5020/mock_data/Current_FEEs_run{current_target[0]}_subrun{current_target[1]}.npy'):
            print('\nand it is ready for analysis')
        else:
            print('\nbut it is not ready for analysis')
            ready=False
            while ready==False:
                ready=os.path.exists(f'/data/user/fbivens5020/mock_data/Current_FEEs_run{current_target[0]}_subrun{current_target[1]}.npy')
            print('\nit is now ready for analysis')
    else:
        print(f'no r0 file for {current_target} found')
        ready=False
        while ready==False:
            ready=os.path.exists(f'/data/user/fbivens5020/mock_data/run{current_target[0]}_subrun{current_target[1]}_r0.tio')
        print('\nfile has been found')
        continue
    
    if current_target[1]==0:
        subruns=[]
        run_data=[]
        sorted_run_data=sorted_run_data_format
        runs.append(current_target[0])
    
    subruns.append(current_target[1])

    print(f'\ndoing all the things and such for run {current_target[0]} subrun {current_target[1]}')

    tim_n=time.time()

    r0_file=r0_file_location+'run'+str(current_target[0])+'_subrun'+str(current_target[1])+'_r0.tio'
    tcal_file=pedestal_path
    r1_file=new_r1_file_location+'run'+str(current_target[0])+'_subrun'+str(current_target[1])+'.r1'

    reader=get_reader(r0_file, tcal_file, r1_file) #get reader data
    time_s=time.time()
    sr_data=collect_stats(reader) #get useable stats from reader
    cuts=get_cuts() #get histogram cuts
    total_run_data.append(sr_data)#puts data for newest sr in the list for the whole run with shape [subrun][type of metric][event]

    sort=real_new_sort(sr_data, current_target[1], sorted_run_data, cuts) #new sorting function it should spit out a nice big list with all relevant data

    sorted_run_data=sort[0]
    sorted_subrun=sort[1]
    sorted_run_array=[]
    for type in range(8):
        sorted_run_array.append(np.array(sorted_run_data[type]))
    sorted_subrun_array=[]
    for type in range(8):
        sorted_subrun_array.append(np.array(sorted_subrun[type]))

    print(f'\nEvents: {len(sorted_subrun_array[0][0])}, showers: {len(sorted_subrun_array[1][0])}, flashers: {len(sorted_subrun_array[2][0])}, noise: {len(sorted_subrun_array[3][0])}')

    print(f'\nsorting the data took {time.time()-time_s} s')
        #event rate histograms here
    time_h=time.time()
    event_rate_hists(current_target, sorted_run_array, sorted_subrun_array, resolution, time_step, display_plots_path, plots_save_path, extra_lines=extra_lines, subrun_plots=subrun_plots)#event rate histograms for overall run and subrun
    print(f'\n event rate histograms took {time.time()-time_h} s\n')
    #2d histogram function here
    time_ht=time.time()
    sorting_hists_2d(cuts, current_target, sorted_run_array, sr_data, display_plots_path, plots_save_path, subrun_plots=subrun_plots, boxes=boxes, regions=noise_shower_regions, flashers=flasher_regions, tight=tight_windows)
    print(f'\n 2d histograms took {time.time()-time_ht} s\n')
    #1d histogram function
    time_ho=time.time()
    sorting_hists_1d(cuts, current_target, sorted_run_array, sr_data, display_plots_path, plots_save_path, subrun_plots=subrun_plots, boxes=boxes, regions=noise_shower_regions, flashers=flasher_regions)
    print(f'\n 1d histograms took {time.time()-time_ho} s\n')
    #physical metrics graph function
    time_p=time.time()
    physical_summary(current_target, total_run_data, physical_metrics_location, display_plots_path, plots_save_path, time_step, modules=modules)
    print(f'\n physical metrics took {time.time()-time_p} s\n')
    #'heat' maps/camera visualizations function here
    

    print(f'\nruns covered: {runs}\nsubruns covered: {subruns}')

    if current_target==final_subrun:
        print('\nfinal subrun reached, ending monitoring')
        break

    if os.path.exists(f'/data/user/fbivens5020/mock_data/run{current_target[0]+1}_subrun{0}_r0.tio'):
        current_target[0]+=1
        current_target[1]=0
        print('\nnew run detected')
        continue

    current_target[1]+=1
    print('\nrestarting loop')


       

