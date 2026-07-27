import target_io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import matplotlib.colors as colors
import matplotlib.patches as patches
from numba import njit
import time

plt.rcParams['figure.dpi'] = 300

#config file loader
def load_config(filepath):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
    return result

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

# #all options and changes listed here:
# r0_file_location='/data/user/fbivens5020/mock_data/'#'folder where all the r0 files are'
# pedestal_path='/data/wipac/CTA/targetcdata/run400032_pedestal.tcal'#'the chosen pedestal'
# new_r1_file_location='/data/user/fbivens5020/mock_data/'#folder where live monitoring r1 files go'
# old_r1_file_location=None #theoretically if you want it to use existing r1 files, option is currently broken
# physical_metrics_location='/data/user/fbivens5020/mock_data/'#"folder where temps currents etc are stored, assuming they're all together"
# modules=22 #the number of operable modules on the camera
# type_number=10 #relates to how how sorting works, don't touch this # no longer in use

# monitoring=False #run the loop
# live=True #is data being taken live, if true will detect the most recent run and start looking at the next run after that
# run_base=400214 #base for function that finds most recent run. only needs to be specific if there aren't any earlier runs in the file
# initial_subrun=[400215,0] #first run and subrun to look at for existing data
# final_subrun=[400215,15] #last subrun, can be used as a stopping point for live monitoring and looking at existing data

# histograms_1d=True #true/false, will 1d histograms be generated at all
# histograms_2d=True #true/false, will 2d histograms be generated at all
# subrun_plots=True #true/false decides if plots will be generated and saved for all subruns as well (noncumulative)
# boxes=True #true/false, will the cut boxes be visible on the histograms
# noise_shower_regions=True #true/false, plots for the noise/shower region will be generated
# flasher_regions=True #true/false, plots for flasher regions will be generated
# tight_windows=True #true/false if false the region graphs look at the whole sorting box, if true they look at a tighter region for more detail between showers and flashers

# extra_lines=False #true/false the lines for showers according to charge std and time std separately will be shown
# resolution=1 #modifier on histogram bin sizes, 1 is a bin per second, 5 is a bin per fifth of a second, etc. it's set up to leave rate invariant
# time_step=60E9 #modifier for the time scale, 60 billion is to take ns to min
# fontsize=14

# subrun_plots=True #true/false decides if versions of every graph are made for each subrun as well
# display_plots_path="/data/user/fbivens5020/DQM_scripts/DQM_plots/display_plots/"# path to folder of display files, these are the ones being overwritten through the loop
# plots_save_path= "/data/user/fbivens5020/DQM_scripts/DQM_plots/subrun_plots/" #place to save all generated plot files for each run and subrun

#returns the sr_data object which is sr[0]: all wfs, sr[1]: mean time, sr[1][ev]: mean time for an event, sr[2]: time std
#sr[3]: charge mean, sr[4]: charge std, sr[5]: event time

# def get_cuts(): #being actively used though i really want to change how this one works
#     charge_mean_shower_max=2000
#     charge_mean_shower_min=40

#     charge_std_shower_max=2000
#     charge_std_shower_min=40

#     charge_mean_flasher_max=3750
#     charge_mean_flasher_min=2000

#     charge_std_flasher_max=1750
#     charge_std_flasher_min=400

#     time_std_shower_max=21
#     time_std_shower_min=14

#     time_std_flasher_max=18
#     time_std_flasher_min=12
    
#     return charge_mean_shower_min, charge_mean_shower_max, charge_std_shower_min, charge_std_shower_max, charge_mean_flasher_min, charge_mean_flasher_max, charge_std_flasher_min, charge_std_flasher_max, time_std_shower_min, time_std_shower_max, time_std_flasher_min, time_std_flasher_max

# def newest_cuts():
#     charge_mean_shower_min=40
#     charge_mean_flasher_min=2500
#     charge_std_shower_min=40
#     charge_std_flasher_max=1250
#     time_std_shower_max=21
#     time_std_flasher_max=17

#     return charge_mean_shower_min, charge_mean_flasher_min, charge_std_shower_min, charge_std_flasher_max, time_std_shower_max, time_std_flasher_max
# #establishes ranges for sorting boxes, make sure to have cuts=get_cuts
#sorting function, to be phased out but still works
# def sort_data(sr_data, cuts, list=False): #Out
#     ch_showers=[]
#     t_showers=[]
#     ch_flashers=[]
#     t_flashers=[]
#     ch_noise=[]
#     t_noise=[]
#     con_showers=[]
#     con_flashers=[]
#     con_noise=[]
#     all_events=[]

#     for ev in range(len(sr_data[0])):

#         all_events.append(ev)

#         if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[4][ev]>cuts[2] and sr_data[4][ev]<cuts[3]:
#           ch_showers.append(ev)
#         elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:# and sr_data[4][ev]>cuts[6] and sr_data[4][ev]<cuts[7]:
#             ch_flashers.append(ev)
#         else:
#             ch_noise.append(ev)

#         if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[2][ev]>cuts[8] and sr_data[2][ev]<cuts[9]:
#             t_showers.append(ev)
#         elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:# and sr_data[2][ev]>cuts[10] and sr_data[2][ev]<cuts[11]:
#             t_flashers.append(ev)
#         else: 
#             t_noise.append(ev)
#     for eve in ch_showers:
#         if eve in t_showers:
#             con_showers.append(eve)
#     for eve in ch_flashers:
#         if eve in t_flashers:
#             con_flashers.append(eve)
#     for eve in ch_noise:
#         if eve in t_noise:
#             con_noise.append(eve)


#     all_events_data=np.zeros((6,len(all_events))) 
#     for ind, ev in enumerate(all_events):
#         all_events_data[0][ind]=ev
#         all_events_data[1][ind]=sr_data[5][ev]

#     charge_showers=np.zeros((6,len(ch_showers)))
#     for ind, ev in enumerate(ch_showers):
#         charge_showers[0][ind]=ev
#         charge_showers[1][ind]=sr_data[5][ev]
    
#     charge_flashers=np.zeros((6,len(ch_flashers)))
#     for ind, ev in enumerate(ch_flashers):
#         charge_flashers[0][ind]=ev
#         charge_flashers[1][ind]=sr_data[5][ev]

#     charge_noise=np.zeros((6,len(ch_noise)))
#     for ind, ev in enumerate(ch_noise):
#         charge_noise[0][ind]=ev
#         charge_noise[1][ind]=sr_data[5][ev]

#     time_showers=np.zeros((6,len(t_showers)))
#     for ind, ev in enumerate(t_showers):
#         time_showers[0][ind]=ev
#         time_showers[1][ind]=sr_data[5][ev]

#     time_flashers=np.zeros((6,len(t_flashers)))
#     for ind, ev in enumerate(t_flashers):
#         time_flashers[0][ind]=ev
#         time_flashers[1][ind]=sr_data[5][ev]

#     time_noise=np.zeros((6,len(t_noise)))
#     for ind, ev in enumerate(t_noise):
#         time_noise[0][ind]=ev
#         time_noise[1][ind]=sr_data[5][ev]

#     conf_showers=np.zeros((6,len(con_showers)))
#     for ind, ev in enumerate(con_showers):
#         conf_showers[0][ind]=ev
#         conf_showers[1][ind]=sr_data[5][ev]

#     conf_flashers=np.zeros((6,len(con_flashers)))
#     for ind, ev in enumerate(con_flashers):
#         conf_flashers[0][ind]=ev
#         conf_flashers[1][ind]=sr_data[5][ev]

#     conf_noise=np.zeros((6,len(con_noise)))
#     for ind, ev in enumerate(con_noise):
#         conf_noise[0][ind]=ev
#         conf_noise[1][ind]=sr_data[5][ev]
        
#     if list==True:
#        print('Showers:',len(con_showers),'\nFlahsers:', len(con_flashers), '\nNoise:', len(con_noise), '\nCharge Showers:',len(ch_showers),'\nCharge Flashers:', len(ch_flashers),'\nCharge Noise:',len(ch_noise),"\nTime Showers:",len(t_showers),'\nTime Noise',len(t_noise))
       
#     return all_events_data, conf_showers, conf_flashers, conf_noise, charge_showers, time_showers, charge_flashers, time_flashers, charge_noise, time_noise
    
# #sorts the data into 9 lists, should be used to create the sorted_data object which has 9 sections with 2 indexes each

#new awesome sorting function in use, is only broken in a couple of ways
# def real_new_sort(sr_data, subrun, sorted_run_data, cuts, subruns):
#    confirmations=[[],[],[],[],0]
#    sorted_subrun=[[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]]]
#    for ev in range(len(sr_data[0])):
        
#         #all events data
#         if subrun==0 and ev==0:
#             subruns.append([subrun, sr_data[5][ev]]) #if first subrun of run record id and starting time
#             sorted_run_data[0][0].append(sr_data[5][ev]-subruns[0][1]) #set first time to 0
#             sorted_run_data[0][7].append(0) #set first dt to zero
#             sorted_subrun[0][0].append(sr_data[5][ev]-subruns[-1][1]) #set starting subrun time to 0
#             sorted_subrun[0][7].append(0) #set first dt of subrun to 0
#         elif ev==0:
#             sorted_run_data[0][0].append(sr_data[5][ev]-subruns[0][1]) # do event time as normal
#             sorted_run_data[0][7].append(sorted_run_data[0][0][-1]-sorted_run_data[0][0][-2]) #do time dt as normal
#             #start of just for subrun stuff
#             subruns.append([subrun, sr_data[5][ev]]) #if first event of subrun record id and starting time
#             sorted_subrun[0][0].append(sr_data[5][ev]-subruns[-1][1]) #set starting subrun time to 0
#             sorted_subrun[0][7].append(0) #set first dt of subrun to 0
#         else:
#             sorted_run_data[0][0].append(sr_data[5][ev]-subruns[0][1]) #do event time like normal
#             sorted_run_data[0][7].append(sorted_run_data[0][0][-1]-sorted_run_data[0][0][-2]) #delta t, this could be done with sr data directly i'm doing it like this for consistency
#             sorted_subrun[0][0].append(sr_data[5][ev]-subruns[-1][1]) #subrun event time like normal
#             sorted_subrun[0][7].append(sorted_subrun[0][0][-1]-sorted_subrun[0][0][-2]) #do subrun dt like normal
            
#         sorted_run_data[0][1].append(sr_data[3][ev]) #mean charge
#         sorted_run_data[0][2].append(sr_data[4][ev]) #charge std
#         sorted_run_data[0][3].append(sr_data[1][ev]) #mean time
#         sorted_run_data[0][4].append(sr_data[2][ev]) #time std
#         sorted_run_data[0][5].append(ev)#event id inside subrun
#         sorted_run_data[0][6].append(subrun)# subrun id to make event id usable
#         #again but for the subrun only
#         sorted_subrun[0][1].append(sr_data[3][ev]) #mean charge
#         sorted_subrun[0][2].append(sr_data[4][ev]) #charge std
#         sorted_subrun[0][3].append(sr_data[1][ev]) #mean time
#         sorted_subrun[0][4].append(sr_data[2][ev]) #time std
#         sorted_subrun[0][5].append(ev)#event id inside subrun
#         sorted_subrun[0][6].append(subrun)# subrun id to make event id usable

#         if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[4][ev]>cuts[2] and sr_data[4][ev]<cuts[3]:
#             #charge showers
#             confirmations[0].append(ev)
#             sorted_run_data[4][0].append(sr_data[5][ev]-subruns[0][1]) #event time corrected
#             sorted_run_data[4][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[4][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[4][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[4][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[4][5].append(ev)#event id inside subrun
#             sorted_run_data[4][6].append(subrun)# subrun id to make event id usable
#             #again but for the subrun only
#             sorted_subrun[4][0].append(sr_data[5][ev]-subruns[-1][1]) #event time corrected
#             sorted_subrun[4][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[4][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[4][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[4][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[4][5].append(ev)#event id inside subrun
#             sorted_subrun[4][6].append(subrun)# subrun id to make event id usable
            
#             if len(sorted_run_data[4][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[4][7].append(0)
#             else:
#                 sorted_run_data[4][7].append(sorted_run_data[4][0][-1]-sorted_run_data[4][0][-2]) #delta t
#             if len(sorted_subrun[4][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[4][7].append(0)
#             else:
#                 sorted_subrun[4][7].append(sorted_subrun[4][0][-1]-sorted_subrun[4][0][-2]) #delta t
          
#         elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:
#             #actually just flashers now
#             sorted_run_data[2][0].append(sr_data[5][ev]-subruns[0][1]) #event time
#             sorted_run_data[2][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[2][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[2][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[2][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[2][5].append(ev)#event id inside subrun
#             sorted_run_data[2][6].append(subrun)# subrun id to make event id usable

#             #again but for the subrun only
#             sorted_subrun[2][0].append(sr_data[5][ev]-subruns[-1][1]) #event time
#             sorted_subrun[2][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[2][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[2][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[2][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[2][5].append(ev)#event id inside subrun
#             sorted_subrun[2][6].append(subrun)# subrun id to make event id usable
            
#             if len(sorted_run_data[2][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[2][7].append(0)
#             else:
#                 sorted_run_data[2][7].append(sorted_run_data[2][0][-1]-sorted_run_data[2][0][-2]) #delta t
#             if len(sorted_subrun[2][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[2][7].append(0)
#             else:
#                 sorted_subrun[2][7].append(sorted_subrun[2][0][-1]-sorted_subrun[2][0][-2]) #delta t
#         else:
#             #charge noise i guess we're still doing this
#             confirmations[1].append(ev)
#             sorted_run_data[6][0].append(sr_data[5][ev]-subruns[0][1]) #event time
#             sorted_run_data[6][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[6][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[6][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[6][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[6][5].append(ev)#event id inside subrun
#             sorted_run_data[6][6].append(subrun)# subrun id to make event id usable

#             #again but for the subrun only
#             sorted_subrun[6][0].append(sr_data[5][ev]-subruns[-1][1]) #event time
#             sorted_subrun[6][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[6][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[6][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[6][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[6][5].append(ev)#event id inside subrun
#             sorted_subrun[6][6].append(subrun)# subrun id to make event id usable

#             if len(sorted_run_data[6][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[6][7].append(0)
#             else:
#                 sorted_run_data[6][7].append(sorted_run_data[6][0][-1]-sorted_run_data[6][0][-2]) #delta t
#             if len(sorted_subrun[6][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[6][7].append(0)
#             else:
#                 sorted_subrun[6][7].append(sorted_subrun[6][0][-1]-sorted_subrun[6][0][-2]) #delta t

#         if sr_data[3][ev]>cuts[0] and sr_data[3][ev]<cuts[1] and sr_data[2][ev]>cuts[8] and sr_data[2][ev]<cuts[9]:
#             #time showers
#             confirmations[2].append(ev)
#             sorted_run_data[5][0].append(sr_data[5][ev]-subruns[0][1]) #event time
#             sorted_run_data[5][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[5][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[5][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[5][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[5][5].append(ev)#event id inside subrun
#             sorted_run_data[5][6].append(subrun)# subrun id to make event id usable

#             #again but for the subrun only
#             sorted_subrun[5][0].append(sr_data[5][ev]-subruns[-1][1]) #event time
#             sorted_subrun[5][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[5][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[5][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[5][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[5][5].append(ev)#event id inside subrun
#             sorted_subrun[5][6].append(subrun)# subrun id to make event id usable

#             if len(sorted_run_data[5][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[5][7].append(0)
#             else:
#                 sorted_run_data[5][7].append(sorted_run_data[5][0][-1]-sorted_run_data[5][0][-2]) #delta t
#             if len(sorted_subrun[5][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[5][7].append(0)
#             else:
#                 sorted_subrun[5][7].append(sorted_subrun[5][0][-1]-sorted_subrun[5][0][-2]) #delta t

#         elif sr_data[3][ev]>cuts[4] and sr_data[3][ev]<cuts[5]:
#             #don't gotta do anything :)
#             confirmations[4]+=1
#         else: 
#             #time noise
#             confirmations[3].append(ev)
#             sorted_run_data[7][0].append(sr_data[5][ev]-subruns[0][1]) #event time
#             sorted_run_data[7][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[7][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[7][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[7][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[7][5].append(ev)#event id inside subrun
#             sorted_run_data[7][6].append(subrun)# subrun id to make event id usable

#             #again but for the subrun only
#             sorted_subrun[7][0].append(sr_data[5][ev]-subruns[-1][1]) #event time
#             sorted_subrun[7][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[7][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[7][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[7][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[7][5].append(ev)#event id inside subrun
#             sorted_subrun[7][6].append(subrun)# subrun id to make event id usable

#             if len(sorted_run_data[7][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[7][7].append(0)
#             else:
#                 sorted_run_data[7][7].append(sorted_run_data[7][0][-1]-sorted_run_data[7][0][-2]) #delta t
#             if len(sorted_subrun[7][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[7][7].append(0)
#             else:
#                 sorted_subrun[7][7].append(sorted_subrun[7][0][-1]-sorted_subrun[7][0][-2]) #delta t

#         if ev in confirmations[0] and confirmations[2]:
#             #confirmed showers
#             sorted_run_data[1][0].append(sr_data[5][ev]-subruns[0][1]) #event time
#             sorted_run_data[1][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[1][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[1][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[1][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[1][5].append(ev)#event id inside subrun
#             sorted_run_data[1][6].append(subrun)# subrun id to make event id usable

#             #again but for the subrun only
#             sorted_subrun[1][0].append(sr_data[5][ev]-subruns[-1][1]) #event time
#             sorted_subrun[1][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[1][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[1][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[1][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[1][5].append(ev)#event id inside subrun
#             sorted_subrun[1][6].append(subrun)# subrun id to make event id usable

#             if len(sorted_run_data[1][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[1][7].append(0)
#             else:
#                 sorted_run_data[1][7].append(sorted_run_data[1][0][-1]-sorted_run_data[1][0][-2]) #delta t
#             if len(sorted_subrun[1][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[1][7].append(0)
#             else:
#                 sorted_subrun[1][7].append(sorted_subrun[1][0][-1]-sorted_subrun[1][0][-2]) #delta t
        
#         if ev in confirmations[1] and confirmations[3]:
#             #confirmed noise
#             sorted_run_data[3][0].append(sr_data[5][ev]-subruns[0][1]) #event time
#             sorted_run_data[3][1].append(sr_data[3][ev]) #mean charge
#             sorted_run_data[3][2].append(sr_data[4][ev]) #charge std
#             sorted_run_data[3][3].append(sr_data[1][ev]) #mean time
#             sorted_run_data[3][4].append(sr_data[2][ev]) #time std
#             sorted_run_data[3][5].append(ev)#event id inside subrun
#             sorted_run_data[3][6].append(subrun)# subrun id to make event id usable

#             #again but for the subrun only
#             sorted_subrun[3][0].append(sr_data[5][ev]-subruns[-1][1]) #event time
#             sorted_subrun[3][1].append(sr_data[3][ev]) #mean charge
#             sorted_subrun[3][2].append(sr_data[4][ev]) #charge std
#             sorted_subrun[3][3].append(sr_data[1][ev]) #mean time
#             sorted_subrun[3][4].append(sr_data[2][ev]) #time std
#             sorted_subrun[3][5].append(ev)#event id inside subrun
#             sorted_subrun[3][6].append(subrun)# subrun id to make event id usable

#             if len(sorted_run_data[3][7])==0: #if first event of type, time dt=0
#                 sorted_run_data[3][7].append(0)
#             else:
#                 sorted_run_data[3][7].append(sorted_run_data[3][0][-1]-sorted_run_data[3][0][-2]) #delta t
#             if len(sorted_subrun[3][7])==0: #if first event of type, time dt=0 for subrun
#                 sorted_subrun[3][7].append(0)
#             else:
#                 sorted_subrun[3][7].append(sorted_subrun[3][0][-1]-sorted_subrun[3][0][-2]) #delta t

#    print(confirmations[4])
#    return sorted_run_data, sorted_subrun

#this new sort is the one, i'm going to delete the others i swear
def another_new_sort(sr_data, subrun, sorted_run_data, config_dict, subruns):
    sorted_subrun=[[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]]]

    flasher_min=int(config_dict["charge_mean_flasher_min"])
    shower_intercept=int(config_dict["shower_intercept"])
    for event in range(len(sr_data[0])):
        #all events data
            
        sorted_run_data[0][0].append(sr_data[5][event]-subruns[0][1]) #set actual time
        sorted_run_data[0][1].append(sr_data[3][event]) #mean charge
        sorted_run_data[0][2].append(sr_data[4][event]) #charge std
        sorted_run_data[0][3].append(sr_data[1][event]) #mean time
        sorted_run_data[0][4].append(sr_data[2][event]) #time std
        sorted_run_data[0][5].append(event)#event id inside subrun
        sorted_run_data[0][6].append(subrun)# subrun id to make event id usable
        #again but for the subrun only
        sorted_subrun[0][0].append(sr_data[5][event]-sr_data[5][0]) #set starting subrun time to 0
        sorted_subrun[0][1].append(sr_data[3][event]) #mean charge
        sorted_subrun[0][2].append(sr_data[4][event]) #charge std
        sorted_subrun[0][3].append(sr_data[1][event]) #mean time
        sorted_subrun[0][4].append(sr_data[2][event]) #time std
        sorted_subrun[0][5].append(event)#event id inside subrun
        sorted_subrun[0][6].append(subrun)# subrun id to make event id usable

        if event==0:
            sorted_run_data[0][7].append(np.nan) #set first dt to zero
            sorted_subrun[0][7].append(np.nan) #set first dt of subrun to 0
        else:
            sorted_run_data[0][7].append(sr_data[5][event]-sr_data[5][event-1]) #set dt
            sorted_subrun[0][7].append(sr_data[5][event]-sr_data[5][event-1]) #set dt for subrun
        #flashers now
        if sr_data[3][event]>flasher_min:

            sorted_run_data[2][0].append(sr_data[5][event]-subruns[0][1]) #set actual time
            sorted_run_data[2][1].append(sr_data[3][event]) #mean charge
            sorted_run_data[2][2].append(sr_data[4][event]) #charge std
            sorted_run_data[2][3].append(sr_data[1][event]) #mean time
            sorted_run_data[2][4].append(sr_data[2][event]) #time std
            sorted_run_data[2][5].append(event)#event id inside subrun
            sorted_run_data[2][6].append(subrun)# subrun id to make event id usable
            #again but for the subrun only
            sorted_subrun[2][0].append(sr_data[5][event]-sr_data[5][0]) #set starting subrun time to 0
            sorted_subrun[2][1].append(sr_data[3][event]) #mean charge
            sorted_subrun[2][2].append(sr_data[4][event]) #charge std
            sorted_subrun[2][3].append(sr_data[1][event]) #mean time
            sorted_subrun[2][4].append(sr_data[2][event]) #time std
            sorted_subrun[2][5].append(event)#event id inside subrun
            sorted_subrun[2][6].append(subrun)# subrun id to make event id usable

            if len(sorted_subrun[2][7])==0:
                sorted_run_data[2][7].append(np.nan)
                sorted_subrun[2][7].append(np.nan)
            else:
                sorted_run_data[2][7].append(sorted_run_data[2][0][-1]-sorted_run_data[2][0][-2])
                sorted_subrun[2][7].append(sorted_subrun[2][0][-1]-sorted_subrun[2][0][-2])

            #if not flasher check shower
        elif sr_data[4][event]>(-sr_data[3][event]+shower_intercept):
            sorted_run_data[1][0].append(sr_data[5][event]-subruns[0][1]) #set actual time
            sorted_run_data[1][1].append(sr_data[3][event]) #mean charge
            sorted_run_data[1][2].append(sr_data[4][event]) #charge std
            sorted_run_data[1][3].append(sr_data[1][event]) #mean time
            sorted_run_data[1][4].append(sr_data[2][event]) #time std
            sorted_run_data[1][5].append(event)#event id inside subrun
            sorted_run_data[1][6].append(subrun)# subrun id to make event id usable
            #again but for the subrun only
            sorted_subrun[1][0].append(sr_data[5][event]-sr_data[5][0]) #set starting subrun time to 0
            sorted_subrun[1][1].append(sr_data[3][event]) #mean charge
            sorted_subrun[1][2].append(sr_data[4][event]) #charge std
            sorted_subrun[1][3].append(sr_data[1][event]) #mean time
            sorted_subrun[1][4].append(sr_data[2][event]) #time std
            sorted_subrun[1][5].append(event)#event id inside subrun
            sorted_subrun[1][6].append(subrun)# subrun id to make event id usable

            if len(sorted_subrun[1][7])==0:
                sorted_run_data[1][7].append(np.nan)
                sorted_subrun[1][7].append(np.nan)
            else:
                sorted_run_data[1][7].append(sorted_run_data[1][0][-1]-sorted_run_data[1][0][-2])
                sorted_subrun[1][7].append(sorted_subrun[1][0][-1]-sorted_subrun[1][0][-2])

            #if not flasher or shower background
        else:
            sorted_run_data[3][0].append(sr_data[5][event]-subruns[0][1]) #set actual time
            sorted_run_data[3][1].append(sr_data[3][event]) #mean charge
            sorted_run_data[3][2].append(sr_data[4][event]) #charge std
            sorted_run_data[3][3].append(sr_data[1][event]) #mean time
            sorted_run_data[3][4].append(sr_data[2][event]) #time std
            sorted_run_data[3][5].append(event)#event id inside subrun
            sorted_run_data[3][6].append(subrun)# subrun id to make event id usable
            #again but for the subrun only
            sorted_subrun[3][0].append(sr_data[5][event]-sr_data[5][0]) #set starting subrun time to 0
            sorted_subrun[3][1].append(sr_data[3][event]) #mean charge
            sorted_subrun[3][2].append(sr_data[4][event]) #charge std
            sorted_subrun[3][3].append(sr_data[1][event]) #mean time
            sorted_subrun[3][4].append(sr_data[2][event]) #time std
            sorted_subrun[3][5].append(event)#event id inside subrun
            sorted_subrun[3][6].append(subrun)# subrun id to make event id usable

            if len(sorted_subrun[3][7])==0:
                sorted_run_data[3][7].append(np.nan)
                sorted_subrun[3][7].append(np.nan)
            else:
                sorted_run_data[3][7].append(sorted_run_data[3][0][-1]-sorted_run_data[3][0][-2])
                sorted_subrun[3][7].append(sorted_subrun[3][0][-1]-sorted_subrun[3][0][-2])

    return sorted_subrun

#event rate histograms, being actively used
def event_rate_hists(current_sr, sorted_run_array, sorted_subrun_array, config_dict):
    
    modifier=float(config_dict["resolution"])
    time_step=float(config_dict["time_step"])
    display_plots_path=config_dict["display_plots_path"]
    plots_save_path=config_dict["plots_save_path"]
    subrun_plots=bool(int(config_dict["subrun_plots"]))
    fontsize=int(config_dict["fontsize"])
    errors=bool(int(config_dict["errors"]))

    fig, ax = plt.subplots()
    all_hist=np.histogram(sorted_run_array[0][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_array[0][0]))], bins = np.arange(sorted_run_data[0][0][0]/(time_step), sorted_run_data[0][0][-1]/(time_step), (1E9/time_step)/modifier))
    show_hist=np.histogram(sorted_run_array[1][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_array[1][0]))], bins = np.arange(sorted_run_array[1][0][0]/(time_step), sorted_run_array[1][0][-1]/(time_step), (1E9/time_step)/modifier))
    flash_hist=np.histogram(sorted_run_array[2][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_array[2][0]))], bins = np.arange(sorted_run_array[2][0][0]/(time_step), sorted_run_array[2][0][-1]/(time_step), (1E9/time_step)/modifier))
    other_hist=np.histogram(sorted_run_array[3][0]/(time_step), weights = [modifier for _ in range(len(sorted_run_array[3][0]))], bins = np.arange(sorted_run_array[3][0][0]/(time_step), sorted_run_array[3][0][-1]/(time_step), (1E9/time_step)/modifier))

    if errors==True:
        ax.errorbar(all_hist[1][:-1],all_hist[0],yerr=np.sqrt(all_hist[0]), label = 'All', marker='o', markersize=3, linestyle='none')
        ax.errorbar(show_hist[1][:-1],show_hist[0],yerr=np.sqrt(show_hist[0]), label = 'Showers', marker='o', markersize=3, linestyle='none')
        ax.errorbar(flash_hist[1][:-1],flash_hist[0],yerr=np.sqrt(flash_hist[0]), label = 'Flashers', marker='o', markersize=3, linestyle='none')
        ax.errorbar(other_hist[1][:-1],other_hist[0],yerr=np.sqrt(other_hist[0]), label = 'Other', marker='o', markersize=3, linestyle='none')
    else:
        ax.errorbar(all_hist[1][:-1],all_hist[0], label = 'All', marker='o', markersize=3, linestyle='none')
        ax.errorbar(show_hist[1][:-1],show_hist[0], label = 'Showers', marker='o', markersize=3, linestyle='none')
        ax.errorbar(flash_hist[1][:-1],flash_hist[0], label = 'Flashers', marker='o', markersize=3, linestyle='none')
        ax.errorbar(other_hist[1][:-1],other_hist[0], label = 'Other', marker='o', markersize=3, linestyle='none')

    ax.legend(loc='upper left', fontsize=fontsize)
    ax.set_title(f"Event Rates, Run {current_sr[0]}, Subruns 0-{current_sr[1]}", fontsize=fontsize)
    ax.set_xlabel("Time [min]", fontsize=fontsize)
    ax.set_ylabel("Rate [Hz]", fontsize=fontsize)
    ax.set_yscale('log')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f"{display_plots_path}event_rate_histogram.jpg", bbox_inches='tight')
    fig.savefig(f"{plots_save_path}run_{current_sr[0]}_event_rate_histogram.jpg", bbox_inches='tight')
    plt.close()
    
    if subrun_plots==True:
        sr_modifier=1/5 #lock subrun plots as having 10 data points per 50 seconds. will make it so freakishly short subruns have no points
        fig, ax = plt.subplots()
        srall_hist=np.histogram(sorted_subrun_array[0][0]/(time_step), weights = [sr_modifier for _ in range(len(sorted_subrun_array[0][0]))], bins = np.arange(sorted_subrun_array[0][0][0]/(time_step), sorted_subrun_array[0][0][-1]/(time_step), (1E9/time_step)/sr_modifier))
        srshow_hist=np.histogram(sorted_subrun_array[1][0]/(time_step), weights = [sr_modifier for _ in range(len(sorted_subrun_array[1][0]))], bins = np.arange(sorted_subrun_array[1][0][0]/(time_step), sorted_subrun_array[1][0][-1]/(time_step), (1E9/time_step)/sr_modifier))
        srflash_hist=np.histogram(sorted_subrun_array[2][0]/(time_step), weights = [sr_modifier for _ in range(len(sorted_subrun_array[2][0]))], bins = np.arange(sorted_subrun_array[2][0][0]/(time_step), sorted_subrun_array[2][0][-1]/(time_step), (1E9/time_step)/sr_modifier))
        srother_hist=np.histogram(sorted_subrun_array[3][0]/(time_step), weights = [sr_modifier for _ in range(len(sorted_subrun_array[3][0]))], bins = np.arange(sorted_subrun_array[3][0][0]/(time_step), sorted_subrun_array[3][0][-1]/(time_step), (1E9/time_step)/sr_modifier))

        if errors==True:
            ax.errorbar(srall_hist[1][:-1],srall_hist[0],yerr=np.sqrt(srall_hist[0]), label = 'All', marker='o', markersize=3, linestyle='none')
            ax.errorbar(srshow_hist[1][:-1],srshow_hist[0],yerr=np.sqrt(srshow_hist[0]), label = 'Showers', marker='o', markersize=3, linestyle='none')
            ax.errorbar(srflash_hist[1][:-1],srflash_hist[0],yerr=np.sqrt(srflash_hist[0]), label = 'Flashers', marker='o', markersize=3, linestyle='none')
            ax.errorbar(srother_hist[1][:-1],srother_hist[0],yerr=np.sqrt(srother_hist[0]), label = 'Other', marker='o', markersize=3, linestyle='none')
        else:
            ax.errorbar(srall_hist[1][:-1],srall_hist[0], label = 'All', marker='o', markersize=3, linestyle='none')
            ax.errorbar(srshow_hist[1][:-1],srshow_hist[0], label = 'Showers', marker='o', markersize=3, linestyle='none')
            ax.errorbar(srflash_hist[1][:-1],srflash_hist[0], label = 'Flashers', marker='o', markersize=3, linestyle='none')
            ax.errorbar(srother_hist[1][:-1],srother_hist[0], label = 'Other', marker='o', markersize=3, linestyle='none')

        ax.legend(loc='upper left', fontsize=fontsize)
        ax.set_title(f"Event Rates, Run {current_sr[0]}, Subrun {current_sr[1]}", fontsize=fontsize)
        ax.set_xlabel("Time [min]", fontsize=fontsize)
        ax.set_ylabel("Rate [Hz]", fontsize=fontsize)
        ax.set_yscale('log')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f"{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_event_rate.jpg", bbox_inches='tight')
        plt.close()

#time dt histograms
def delt_hists(current_sr, sorted_run_data, sorted_subrun, config_dict):
    display_plots_path=config_dict["display_plots_path"]
    plots_save_path=config_dict["plots_save_path"]
    subrun_plots=bool(int(config_dict["subrun_plots"]))
    fontsize=int(config_dict["fontsize"])
    bins=int(config_dict["bins"])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][7], bins = bins, log=True)
    ax.set_title(f"dt, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
    ax.set_xlabel("dt (ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}time_dt_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_dt_histogram.jpg', bbox_inches='tight')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][7], bins = bins, log=True, range=(0,5E5))
    ax.set_title(f"dt, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
    ax.set_xlabel("dt (ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}time_dt_origin_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_dt_origin_histogram.jpg', bbox_inches='tight')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[1][7], bins = bins, log=True)
    ax.set_title(f"dt, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Showers)", fontsize=fontsize)
    ax.set_xlabel("dt (ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}time_dt_shower_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_dt_shower_histogram.jpg', bbox_inches='tight')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[2][7], bins = bins, log=True)
    ax.set_title(f"dt, Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flashers)", fontsize=fontsize)
    ax.set_xlabel("dt (ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}time_dt_flasher_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_dt_flasher_histogram.jpg', bbox_inches='tight')
    plt.close()

    if subrun_plots==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_subrun[0][7], bins = bins, log=True)
        ax.set_title(f"dt, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("dt (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_dt_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_subrun[0][7], bins = bins, log=True, range=(0,5E5))
        ax.set_title(f"dt, Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("dt (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_dt_origin_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_subrun[1][7], bins = bins, log=True)
        ax.set_title(f"dt, Run {current_sr[0]}, Subrun {current_sr[1]} (Showers)", fontsize=fontsize)
        ax.set_xlabel("dt (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_dt_shower_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_subrun[2][7], bins = bins, log=True)
        ax.set_title(f"dt, Run {current_sr[0]}, Subrun {current_sr[1]} (Flashers)", fontsize=fontsize)
        ax.set_xlabel("dt (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_dt_flasher_histogram.jpg', bbox_inches='tight')
        plt.close()

#2d histograms, being actively used
def sorting_hists_2d(current_sr, sorted_run_data, sr_data, config_dict):
   display_plots_path=config_dict["display_plots_path"]
   plots_save_path=config_dict["plots_save_path"]
   subrun_plots=bool(int(config_dict["subrun_plots"]))
   boxes=bool(int(config_dict["boxes"]))
   regions=bool(int(config_dict["shower_regions"]))
   flashers=bool(int(config_dict["flasher_regions"]))
   logscale=bool(int(config_dict["logscale"]))
   fontsize=int(config_dict["fontsize"])
   bins=int(config_dict["bins"])
   flasher_min=int(config_dict["charge_mean_flasher_min"])
   shower_intercept=int(config_dict["shower_intercept"])
   xbounds=[(-40), shower_intercept]
   ybounds=[(40+shower_intercept),0]


   fig=plt.figure()
   ax=fig.add_subplot(111)
   ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][2], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

   if boxes==True:
      ax.vlines(flasher_min,0,1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
      ax.plot(xbounds, ybounds, color='green', label='Shower Cut')
      ax.legend(loc='upper right', fontsize=fontsize)


   ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
   ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
   ax.set_ylabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
   plt.xticks(fontsize=fontsize)
   plt.yticks(fontsize=fontsize)
   fig.savefig(f'{display_plots_path}charge_std_charge_mean_histogram.jpg',bbox_inches='tight')
   fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_histogram.jpg',bbox_inches='tight')
   plt.close()

   fig=plt.figure()
   ax=fig.add_subplot(111)
   ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][4], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

   if boxes==True:
      ax.vlines(flasher_min,0,30, colors='orange', linestyle='dashed', label='Flasher Cut')
      ax.legend(loc='upper right', fontsize=fontsize)

   ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
   ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
   ax.set_ylabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
   plt.xticks(fontsize=fontsize)
   plt.yticks(fontsize=fontsize)
   fig.savefig(f'{display_plots_path}time_std_charge_mean_histogram.jpg', bbox_inches='tight')
   fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_histogram.jpg', bbox_inches='tight')
   plt.close()

   if logscale==True:
       
      charge_mean_log=np.log10(sorted_run_data[0][1])
      charge_std_log=np.log10(sorted_run_data[0][2])
      time_std_log=np.log10(sorted_run_data[0][4])

      xline=np.arange(1,shower_intercept,(shower_intercept/10))
      yline=(xline*(-1))+shower_intercept

      fig=plt.figure()
      ax=fig.add_subplot(111)

      ax.hist2d(charge_mean_log, charge_std_log, bins=bins, range=[[0, np.nanmax(charge_mean_log)+0.5],[np.nanmin(charge_std_log)-0.5, np.nanmax(charge_std_log)+0.5]], cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

      if boxes==True:
        ax.vlines(np.log10(flasher_min),0, np.nanmax(charge_std_log)+0.5, linestyle='dashed', colors='orange', label='Flasher Cut')
        ax.plot(np.log10(xline),np.log10(yline), color='green', label='Shower Cut')
        ax.legend(loc='lower right',fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
      ax.set_xlabel("Log Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Log Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{display_plots_path}charge_std_charge_mean_log_histogram.jpg', bbox_inches='tight')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_log_histogram.jpg', bbox_inches='tight')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)

      ax.hist2d(charge_mean_log, time_std_log, bins=bins, range=[[0, np.nanmax(charge_mean_log)+0.5],[np.nanmin(time_std_log), np.nanmax(time_std_log)]], cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

      if boxes==True:
          ax.vlines(np.log10(flasher_min),0, np.nanmax(time_std_log)+0.2, linestyle='dashed', colors='orange', label='Flasher Cut')
          ax.legend(loc='upper right',fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
      ax.set_xlabel("Log Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Log Peak Time Standard Deviation (ADC*ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{display_plots_path}time_std_charge_mean_log_histogram.jpg', bbox_inches='tight')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_log_histogram.jpg', bbox_inches='tight')
      plt.close()

   if regions==True:
    
      charge_window=[[np.nanmin(sorted_run_data[0][1]),shower_intercept+60],[0,shower_intercept+60]]
      time_window=[[np.nanmin(sorted_run_data[0][1]), shower_intercept+60],[16, np.nanmax(sorted_run_data[0][4])]]
    
      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][2], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=charge_window)

      if boxes==True:
        ax.plot([np.nanmin(sorted_run_data[0][1]),shower_intercept], [(-1)*np.nanmin(sorted_run_data[0][1])+shower_intercept, 0], color='green', label='Shower Cut')
        ax.legend(loc='upper right', fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)", fontsize=fontsize)
      ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{display_plots_path}charge_std_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][4], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=time_window)

    #   if boxes==True: #it really doesn't need this
    #     ax.vlines(flasher_min,0,1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
    #     ax.legend(loc='upper right', fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)", fontsize=fontsize)
      ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{display_plots_path}time_std_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
      plt.close()
   
   if flashers==True:

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][2], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[flasher_min-20,flasher_min+1500],[250, 1500]])
      if boxes==True:
        ax.vlines(flasher_min,0, 1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
        ax.legend(loc='upper right', fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)", fontsize=fontsize)
      ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{display_plots_path}charge_std_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sorted_run_data[0][1], sorted_run_data[0][4], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=[[flasher_min-20,flasher_min+1500],[10,16]])

      if boxes==True:
        ax.vlines(flasher_min,0, 26, colors='orange', linestyle='dashed', label='Flasher Cut')
        ax.legend(loc='upper right', fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)", fontsize=fontsize)
      ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{display_plots_path}time_std_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
      plt.close()

   if subrun_plots==True:
      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sr_data[3], sr_data[4], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

      if boxes==True:
         ax.vlines(flasher_min,0,1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
         ax.plot(xbounds, ybounds, color='green', label='Shower Cut')
         ax.legend(loc='upper right', fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
      ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_charge_mean_histogram.jpg', bbox_inches='tight')
      plt.close()

      fig=plt.figure()
      ax=fig.add_subplot(111)
      ax.hist2d(sr_data[3], sr_data[2], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

      if boxes==True:
         ax.vlines(flasher_min,0,30, colors='orange', linestyle='dashed', label='Flasher Cut')
         ax.legend(loc='upper right', fontsize=fontsize)

      ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
      ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
      ax.set_ylabel("Peak Time Standard (ns)", fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.yticks(fontsize=fontsize)
      fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_histogram.jpg', bbox_inches='tight')
      plt.close()

      if logscale==True:
             
        sr_charge_mean_log=np.log10(sr_data[3])
        sr_charge_std_log=np.log10(sr_data[4])
        sr_time_std_log=np.log10(sr_data[2])
    
        xline=np.arange(1,shower_intercept,(shower_intercept/10))
        yline=(xline*(-1))+shower_intercept
    
        fig=plt.figure()
        ax=fig.add_subplot(111)
    
        ax.hist2d(sr_charge_mean_log, sr_charge_std_log, bins=bins, range=[[0, np.nanmax(sr_charge_mean_log)+0.5],[np.nanmin(sr_charge_std_log)-0.5, np.nanmax(sr_charge_std_log)+0.5]], cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))
    
        if boxes==True:
            ax.vlines(np.log10(flasher_min),0, np.nanmax(sr_charge_std_log)+0.5, linestyle='dashed', colors='orange', label='Flasher Cut')
            ax.plot(np.log10(xline),np.log10(yline), color='green', label='Shower Cut')
            ax.legend(loc='lower right',fontsize=fontsize)
    
        ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("Log Charge Mean (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Log Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_charge_mean_log_histogram.jpg', bbox_inches='tight')
        plt.close()
    
        fig=plt.figure()
        ax=fig.add_subplot(111)
    
        ax.hist2d(sr_charge_mean_log, sr_time_std_log, bins=bins, range=[[0, np.nanmax(sr_charge_mean_log)+0.5],[np.nanmin(sr_time_std_log), np.nanmax(sr_time_std_log)]], cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))
    
        if boxes==True:
            ax.vlines(np.log10(flasher_min),0, np.nanmax(sr_time_std_log)+0.2, linestyle='dashed', colors='orange', label='Flasher Cut')
            ax.legend(loc='upper right',fontsize=fontsize)
    
        ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("Log Charge Mean (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Log Peak Time Standard Deviation (ADC*ns)", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_log_histogram.jpg', bbox_inches='tight')
        plt.close()

      if regions==True:
      
         charge_window=[[np.nanmin(sr_data[3]),shower_intercept+60],[0,shower_intercept+60]]
         time_window=[[np.nanmin(sr_data[3]), shower_intercept+60],[16, np.nanmax(sr_data[2])]]
    
         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[4], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=charge_window)

         if boxes==True:
        #    ax.vlines(flasher_min,0,1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
           ax.plot(xbounds, ybounds, color='green', label='Shower Cut')
           ax.legend(loc='upper right', fontsize=fontsize)

         ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)", fontsize=fontsize)
         ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
         ax.set_ylabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
         plt.xticks(fontsize=fontsize)
         plt.yticks(fontsize=fontsize)
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
         plt.close()

         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[2], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=time_window)

        #  if boxes==True: # doesn't really need it here
        #     ax.vlines(flasher_min,0,1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
        #     ax.legend(loc='upper right', fontsize=fontsize)

         ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)", fontsize=fontsize)
         ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
         ax.set_ylabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
         plt.xticks(fontsize=fontsize)
         plt.yticks(fontsize=fontsize)
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
         plt.close()
         
      if flashers==True:

         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[4], bins = bins,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[flasher_min-20,flasher_min+1500],[250, 1500]])

         if boxes==True:
            ax.vlines(flasher_min,0, 1E5, colors='orange', linestyle='dashed', label='Flasher Cut')
            ax.legend(loc='upper right', fontsize=fontsize)

         ax.set_title(f"Run {current_sr[0]}, Subruns{current_sr[1]} (Flasher Region)", fontsize=fontsize)
         ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
         ax.set_ylabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
         plt.xticks(fontsize=fontsize)
         plt.yticks(fontsize=fontsize)
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
         plt.close()

         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.hist2d(sr_data[3], sr_data[2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None), range=[[flasher_min-20,flasher_min+1500],[10,16]])

         if boxes==True:
            ax.vlines(flasher_min,0, 26, colors='orange', linestyle='dashed', label='Flasher Cut')
            ax.legend(loc='upper right', fontsize=fontsize)

         ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Flasher Region)", fontsize=fontsize)
         ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
         ax.set_ylabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
         plt.xticks(fontsize=fontsize)
         plt.yticks(fontsize=fontsize)
         fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
         plt.close()

#1d histograms, being used
def sorting_hists_1d(current_sr, sorted_run_data, sr_data, config_dict):
    flasher_min=int(config_dict["charge_mean_flasher_min"])
    display_plots_path=config_dict["display_plots_path"]
    plots_save_path=config_dict["plots_save_path"]
    subrun_plots=bool(int(config_dict["subrun_plots"]))
    boxes=bool(int(config_dict["boxes"]))
    regions=bool(int(config_dict["shower_regions"]))
    bins=int(config_dict["bins"])
    fontsize=int(config_dict["fontsize"])
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][1], bins = bins, log=True)

    if boxes==True:
        ax.vlines(flasher_min, 0, 10E5, colors='orange', linestyle='dashed', label='Flasher Cut')

    ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
    ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}charge_mean_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_mean_histogram.jpg', bbox_inches='tight')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][2], bins = bins, log=True)

    ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
    ax.set_xlabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}charge_std_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_histogram.jpg', bbox_inches='tight')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][3], bins = bins, log=True)
    ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
    ax.set_xlabel("Peak Time Mean (ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    fig.savefig(f'{display_plots_path}time_mean_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_mean_histogram.jpg', bbox_inches='tight')
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(sorted_run_data[0][4], bins = bins, log=True)

    ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (All Events)", fontsize=fontsize)
    ax.set_xlabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
    ax.set_ylabel("Number of Events", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}time_std_histogram.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_histogram.jpg', bbox_inches='tight')
    plt.close()

    if regions==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][1], bins = bins, log=True, range=(-40, 400))

        ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)", fontsize=fontsize)
        ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{display_plots_path}charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][1], bins = bins, log=True, range=(flasher_min, flasher_min+1500))

        ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Flasher Region)", fontsize=fontsize)
        ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{display_plots_path}charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][2], bins = bins, log=True, range=(0,400))

        ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)", fontsize=fontsize)
        ax.set_xlabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{display_plots_path}charge_std_shower_region_histogram.jpg', bbox_inches='tight')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_charge_std_shower_region_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sorted_run_data[0][4], bins = bins, log=True, range=(18, 26))

        ax.set_title(f"Run {current_sr[0]}, Subruns 0-{current_sr[1]} (Shower Region)", fontsize=fontsize)
        ax.set_xlabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{display_plots_path}time_std_shower_region_histogram.jpg', bbox_inches='tight')
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_time_std_shower_region_histogram.jpg', bbox_inches='tight')
        plt.close()

    if subrun_plots==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[3], bins = bins, log=True)

        if boxes==True:
            ax.vlines(flasher_min, 0, 10E5, colors='orange', linestyle='dashed', label='Flasher Cut')
            ax.legend(loc='upper right', fontsize=fontsize)

        ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_mean_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[3], bins = bins, log=True, range=(flasher_min, flasher_min+1500))

        ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Flasher Region)", fontsize=fontsize)
        ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_mean_flasher_region_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[4], bins = bins, log=True)

        ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[1], bins = bins, log=True)
        ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("Peak Time Mean (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_mean_histogram.jpg', bbox_inches='tight')
        plt.close()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(sr_data[2], bins = bins, log=True)

        ax.set_title(f"Run {current_sr[0]}, Subruns {current_sr[1]} (All Events)", fontsize=fontsize)
        ax.set_xlabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
        ax.set_ylabel("Number of Events", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_histogram.jpg', bbox_inches='tight')
        plt.close()

        if regions==True:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[3], bins = bins, log=True, range=(-40, 400))

            ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)", fontsize=fontsize)
            ax.set_xlabel("Charge Mean (ADC*ns)", fontsize=fontsize)
            ax.set_ylabel("Number of Events", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_mean_shower_region_histogram.jpg', bbox_inches='tight')
            plt.close()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[4], bins = bins, log=True, range=(0, 400))

            ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)", fontsize=fontsize)
            ax.set_xlabel("Charge Standard Deviation (ADC*ns)", fontsize=fontsize)
            ax.set_ylabel("Number of Events", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_charge_std_shower_region_histogram.jpg', bbox_inches='tight')
            plt.close()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist(sr_data[2], bins = bins, log=True, range=(18,26))

            ax.set_title(f"Run {current_sr[0]}, Subrun {current_sr[1]} (Shower Region)", fontsize=fontsize)
            ax.set_xlabel("Peak Time Standard Deviation (ns)", fontsize=fontsize)
            ax.set_ylabel("Number of Events", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            fig.savefig(f'{plots_save_path}run_{current_sr[0]}_subrun_{current_sr[1]}_time_std_shower_region_histogram.jpg', bbox_inches='tight')
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

def environmental_summary(current_sr, subruns, config_dict):
    modules=int(config_dict["modules"])
    time_step=float(config_dict["time_step"])
    display_plots_path=config_dict["display_plots_path"]
    plots_save_path=config_dict["plots_save_path"]
    fontsize=int(config_dict["fontsize"])

    fpmTemp_list=[]
    feeTemp_list=[]
    hv_list=[]
    current_list=[]

    for sr in range(current_sr[1]+1):

      fpmTemp_data=np.load(config_dict["FPM_temp_file"].format(current_sr[0], sr))
      fpmTemp_list.append(fpmTemp_data)

      feeTemp_data=np.load(config_dict["FEE_temp_file"].format(current_sr[0], sr))
      feeTemp_list.append(feeTemp_data)

      hv_data=np.load(config_dict["hv_file"].format(current_sr[0], sr))
      hv_list.append(hv_data)

      current_data=np.load(config_dict["current_file"].format(current_sr[0], sr))
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
           fpmTemps[quad][3][sr]=subruns[sr][1]

        for board in range(modules*2):
           feeTemps[board][0][sr]=sr
           feeTemps[board][1][sr]=feeTemp_list[sr][board]
           feeTemps[board][2][sr]=board//2
           feeTemps[board][3][sr]=subruns[sr][1]

        for mod in range(modules):
           hv[mod][0][sr]=sr
           hv[mod][1][sr]=hv_list[sr][mod]
           hv[mod][2][sr]=mod
           hv[mod][3][sr]=subruns[sr][1]
           current[mod][0][sr]=sr
           current[mod][1][sr]=current_list[sr][mod]
           current[mod][2][sr]=mod
           current[mod][3][sr]=subruns[sr][1]

    fig, ax,=plt.subplots()
    for quad in range(modules*4):
       ax.plot(fpmTemps[quad][3]/(time_step), fpmTemps[quad][1])
    ax.set_ylabel("FPM Temperature (C)", fontsize=fontsize)
    ax.set_xlabel("Time (min)", fontsize=fontsize)
    ax.set_title('Environmental Metrics', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}FPM_temps_plot.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_FPM_temps_plot.jpg', bbox_inches='tight')
    plt.close()

    fig, ax,=plt.subplots()
    for board in range(modules*2):
       ax.plot(feeTemps[board][3]/(time_step), feeTemps[board][1])
    ax.set_ylabel("FEE Temperature (C)", fontsize=fontsize)
    ax.set_xlabel("Time (min)", fontsize=fontsize)
    ax.set_title('Environmental Metrics', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}FEE_temps_plot.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_FEE_temps_plot.jpg', bbox_inches='tight')
    plt.close()

    fig, ax,=plt.subplots()
    for mod in range(modules):
       ax.plot(hv[mod][3]/(time_step),hv[mod][1])
    ax.set_ylabel("HV (V)", fontsize=fontsize)
    ax.set_xlabel("Time (min)", fontsize=fontsize)
    ax.set_title('Environmental Metrics', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}HV_plot.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_HV_plot.jpg', bbox_inches='tight')
    plt.close()

    fig, ax,=plt.subplots()
    for mod in range(modules):
       ax.plot(current[mod][3]/(time_step),current[mod][1])
    ax.set_ylabel("Current (A)", fontsize=fontsize)
    ax.set_xlabel("Time (min)", fontsize=fontsize)
    ax.set_title('Environmental Metrics', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.savefig(f'{display_plots_path}current_plot.jpg', bbox_inches='tight')
    fig.savefig(f'{plots_save_path}run_{current_sr[0]}_current_plot.jpg', bbox_inches='tight')
    plt.close()

    
#CONDENSED FUNCTION, to be deleted

# def sr_summary(run_id, sr_number, metrics_location, r0_location, tcal_location, r1_location, ev_save_location, phys_save_location, test=False, resolution=1, modules=22):

#     event_rate_summary(run_id, sr_number, r0_location, tcal_location, r1_location, ev_save_location, test=False, resolution=resolution)
#     print("Summary plot B has been generated")

#     physical_summary(run_id, sr_number, metrics_location, phys_save_location, modules=modules)
#     print("Summary plot A has been generated")

#     print(f"Run {run_id} sr{sr_number} summary")

#NEWEST FILE DETECTOR function, is it in use? i don't actually know

# def get_new_sr(physical_metrics_location, run_base=400196, subrun_base=0): #being actively used

#    run=None
#    while run==None:
#       if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{run_base+1}_subrun0.npy")==True:
#          run_base=run_base+1
#       else:
#          run=run_base

#    subrun=None
#    while subrun==None:
#       if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{run}_subrun{subrun_base+1}.npy")==True:
#          subrun_base=subrun_base+1
#       else:
#          subrun=subrun_base
#    return run, subrun

config_dict = load_config("/data/user/fbivens5020/DQM_scripts/DQM_config.txt")
monitoring=bool(config_dict["monitoring"])
live=bool(int(config_dict["live"]))
current_target=[int(config_dict["initial_run"]),0] #if not live the inital run is taken as the starting point
if live==True:
    current_target=[(int(config_dict["initial_run"])+1),0] #if live we're looking for the next run assuming inital run is completed

runs=[] #will hold run ids and starting times
subruns=[] #will hold subrun ids and subrun starting times, will be cleared on each run
sorted_run_data_format=[[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[],[]]]
sorted_run_data=sorted_run_data_format #4 lists of 8 empty lists, will be the sorted data object the plots use
final_subrun=[int(config_dict["final_run"]), 15] #this is sort of a testing object, doesn't work super great across runs
while monitoring==True:
    config_dict = load_config("/data/user/fbivens5020/DQM_scripts/DQM_config.txt")
    print(f'\nlooking at run {current_target[0]}, sub-run {current_target[1]}')
    #loop that should check and wait for the right file for as long as it needs to
    #works across one run, hasn't been tested for switching to a new run
    if os.path.exists(config_dict["r0_file_location"].format(current_target[0],current_target[1])):
        print(f'\nr0 file for {current_target} found')
        if os.path.exists(config_dict["current_file"].format(current_target[0],current_target[1])):
            print('\nand it is ready for analysis')
        else:
            print('\nbut it is not ready for analysis')
            ready=False
            while ready==False:
                ready=os.path.exists(config_dict["current_file"].format(current_target[0],current_target[1]))
            print('\nit is now ready for analysis')
    else:
        print(f'no r0 file for {current_target} found') #loop that should prevent script from going to the next run too early
        ready=False
        set=False
        while ready==False or set==False:
            ready=os.path.exists(config_dict["r0_file_location"].format(current_target[0],current_target[1]))
            set=os.path.exists(config_dict["r0_file_location"].format(current_target[0]+1,0))
        if ready==True:
            print('\nfile has been found')
            continue
        elif set==True:
            print('\nnew run has been found')
            current_target=[current_target[0]+1, 0]
            continue
    
    if current_target[1]==0:
        subruns=[]
        sorted_run_data=sorted_run_data_format
        runs.append(current_target[0])
    
    # subruns.append(current_target[1])

    print(f'\ndoing all the things and such for run {current_target[0]} subrun {current_target[1]}')
    
    time_n=time.time()

    r0_file=config_dict["r0_file_location"].format(current_target[0],current_target[1])
    tcal_file=config_dict["pedestal_path"]
    r1_file=config_dict["r1_file_location"].format(current_target[0],current_target[1])

    reader=get_reader(r0_file, tcal_file, r1_file) #get reader data
    time_s=time.time()
    sr_data=collect_stats(reader) #get useable stats from reader
    subruns.append([current_target[1], sr_data[5][0]])
    sorted_subrun=another_new_sort(sr_data, current_target[1], sorted_run_data, config_dict, subruns) #new sorting function it should spit out a nice big list with all relevant data

    sorted_run_array=[]
    for type in range(4):
        sorted_run_array.append(np.array(sorted_run_data[type]))
    sorted_subrun_array=[]
    for type in range(4):
        sorted_subrun_array.append(np.array(sorted_subrun[type]))

    print(f'\nEvents: {len(sorted_subrun_array[0][0])}, showers: {len(sorted_subrun_array[1][0])}, flashers: {len(sorted_subrun_array[2][0])}, other: {len(sorted_subrun_array[3][0])}')

    print(f'\nsorting the data took {time.time()-time_s} s')
        #event rate histograms here

    time_h=time.time()
    event_rate_hists(current_target, sorted_run_array, sorted_subrun_array, config_dict)#event rate histograms for overall run and subrun
    print(f'\n event rate histograms took {time.time()-time_h} s\n')
    #2d histogram function here
    if bool(int(config_dict["histograms_2d"]))==True:
        time_ht=time.time()
        sorting_hists_2d(current_target, sorted_run_data, sr_data, config_dict)
        print(f'\n 2d histograms took {time.time()-time_ht} s\n')

    #1d histogram function
    if bool(int(config_dict["histograms_1d"]))==True:
        time_ho=time.time()
        sorting_hists_1d(current_target, sorted_run_array, sr_data, config_dict)
        print(f'\n 1d histograms took {time.time()-time_ho} s\n')

    #physical metrics graph function
    time_p=time.time()
    environmental_summary(current_target, subruns, config_dict)
    print(f'\n physical metrics took {time.time()-time_p} s\n')
    #'heat' maps/camera visualizations function here

    #time dt graphs
    if bool(int(config_dict["histograms_dt"]))==True:
        time_dt=time.time()
        delt_hists(current_target, sorted_run_array, sorted_subrun_array, config_dict)
        print(f'\n time dt plots took {time.time()-time_p} s\n')

    print(f'\nsummary of run {current_target[0]} subrun {current_target[1]} took {time.time()-time_n}s\n')

    print(f'\nruns covered: {runs}\nsubruns covered: {subruns}')

    if current_target==final_subrun:
        print('\nfinal subrun reached, ending monitoring')
        break

    current_target[1]+=1
    print('\nrestarting loop')


       

