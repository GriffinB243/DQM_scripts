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
old_r1_file_location='folder where existing r1 files are stored, not currently relevant'
physical_metrics_location='/data/user/fbivens5020/mock_data/'#"folder where temps currents etc are stored, assuming they're all together"
modules="some specific number"

live_monitoring='true or false'
archival_data='true or false'

histograms='true or false'
noise_shower_regions='true or false'
flasher_regions='true or false'
boxes='true or false'

histogram_location_charge='path to save total charge std vs mean charge histograms'
histogram_location_time='path to save total time std vs mean charge histograms'
regions_location_charge='path to save noise/shower region charge std vs mean charge histograms'
regions_location_time='path to save noise/shower region time std vs mean charge histogram'
flasher_location_charge='path to save flasher region charge std vs mean charge histogram'
flasher_location_time='path to save flasher region time std vs mean charge histogram'

extra_lines='true or false'
resolution='some number'
run_base='some more specific number'
summary_plots_location_1="/home/fbivens5020/imgB.jpg"#'path where event rate histogram goes'
summary_plots_location_2="/home/fbivens5020/imgA.jpg" #path where the physical metrics go


#get the reader object with your r0_file_path, chosen pedestal path, and r1_file_path if one exists

def get_reader(r0_path, tcal_path, r1_path):

    tcal_ped_path = tcal_path

    r0_file_path = r0_path

    file_path = r1_path
    
    os.system(f"apply_calibration_SCT -p {tcal_ped_path} -i {r0_file_path} -o {file_path}") # This will calibrate the data and make a r1 file
        
    reader = target_io.WaveformArrayReader(file_path)
    return reader

#start of stat collection sequence

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

def collect_stats(reader):

    all_wfs, timess = read_wfs(None, reader=reader)
    int_charges = get_int_charges(all_wfs, int_win=4)
    peak_times = get_max_time(all_wfs)
    stats_all = get_event_stats(int_charges, peak_times)


    return all_wfs, stats_all[0], stats_all[1], stats_all[2], stats_all[3], timess

#returns the sr_data object which is sr[0]: all wfs, sr[1]: mean time, sr[1][ev]: mean time for an event, sr[2]: time std
#sr[3]: charge mean, sr[4]: charge std, sr[5]: event time

def get_cuts():
    chshowx_max=1250
    chshowx_min=25
    chshowy_max=2000
    chshowy_min=40

    chflashx_max=3750
    chflashx_min=2000
    chflashy_max=1750
    chflashy_min=800

    tshowx_max=1250
    tshowx_min=0
    tshowy_max=21.5
    tshowy_min=14

    tflashx_max=3750
    tflashx_min=2000
    tflashy_max=18
    tflashy_min=12
    return chshowx_max, chshowx_min, chshowy_max, chshowy_min, chflashx_max, chflashx_min, chflashy_max, chflashy_min, tshowx_max, tshowx_min, tshowy_max, tshowy_min, tflashx_max, tflashx_min, tflashy_max, tflashy_min

#establishes ranges for sorting boxes, make sure to have cuts=get_cuts
#
def get_hists(sr_data, cuts, display=True, regions=True, flashers=True, boxes=True):

    if display==True:

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist2d(sr_data[3], sr_data[4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[1],cuts[3]), width=(cuts[0]-cuts[1]), height=(cuts[2]-cuts[3]), linewidth=1, color='green', fill=False))
            ax.add_patch(patches.Rectangle(xy=(cuts[5],cuts[7]), width=(cuts[4]-cuts[5]), height=(cuts[6]-cuts[7]), linewidth=1, color='red', fill=False))

        plt.title("Charge std vs Mean Charge for all events")
        plt.xlabel("Mean charge (ADC*ns)")
        plt.ylabel("Charge std (ADC*ns)")
        plt.show()

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist2d(sr_data[3], sr_data[2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None))

        if boxes==True:
            ax.add_patch(patches.Rectangle(xy=(cuts[9],cuts[11]), width=(cuts[8]-cuts[9]), height=(cuts[10]-cuts[11]), linewidth=1, color='green', fill=False))
            ax.add_patch(patches.Rectangle(xy=(cuts[13],cuts[15]), width=(cuts[12]-cuts[13]), height=(cuts[14]-cuts[15]), linewidth=1, color='red', fill=False))

        plt.title("Time std vs Mean Charge for all events")
        plt.xlabel("Mean charge (ADC*ns)")
        plt.ylabel("Time std (ns)")
        plt.show()


        if regions==True:

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist2d(sr_data[3], sr_data[4], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range = [[cuts[1]-50,cuts[0]+50],[cuts[3]-50,cuts[2]+50]])
            
            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[1],cuts[3]), width=(cuts[0]-cuts[1]), height=(cuts[2]-cuts[3]), linewidth=1, color='green', fill=False))

            plt.title("Charge std vs Mean Charge for all events (Noise/Shower region)")
            plt.xlabel("Mean charge (ADC*ns)")
            plt.ylabel("Charge std (ADC*ns)")
            plt.show()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist2d(sr_data[3], sr_data[2], bins = 400,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[cuts[9]-50,cuts[8]+50],[cuts[11]-3,cuts[10]+3]])

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[9],cuts[11]), width=(cuts[8]-cuts[9]), height=(cuts[10]-cuts[11]), linewidth=1, color='green', fill=False))

            plt.title("Time std vs Mean Charge for all events(Noise/Shower region)")
            plt.xlabel("Mean charge (ADC*ns)")
            plt.ylabel("Time std (ns)")
            plt.show()

        if flashers==True:

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist2d(sr_data[3], sr_data[4], bins = 500,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[cuts[5]-50,cuts[4]+50],[cuts[7]-50,cuts[6]+50]])

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[5],cuts[7]), width=(cuts[4]-cuts[5]), height=(cuts[6]-cuts[7]), linewidth=1, color='red', fill=False))

            plt.title("Charge std vs Mean Charge for all events (Flasher region)")
            plt.xlabel("Mean charge (ADC*ns)")
            plt.ylabel("Charge std (ADC*ns)")
            plt.show()

            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.hist2d(sr_data[3], sr_data[2], bins = 500,cmap=plt.cm.jet ,norm=colors.LogNorm(vmin=1, vmax = None),range=[[cuts[13]-50,cuts[12]+50],[cuts[15]-3,cuts[14]+3]])

            if boxes==True:
                ax.add_patch(patches.Rectangle(xy=(cuts[13],cuts[15]), width=(cuts[12]-cuts[13]), height=(cuts[14]-cuts[15]), linewidth=1, color='red', fill=False))
                
            plt.title("Time std vs Mean Charge for all events(Flasher region)")
            plt.xlabel("Mean charge (ADC*ns)")
            plt.ylabel("Time std (ns)")
            plt.show()

#makes histograms that help determine boxes for the sorting step, can control how many are created.

def sort_data(sr_data, cuts, list=False):
    ch_showers=[]
    t_showers=[]
    ch_flashers=[]
    t_flashers=[]
    ch_noise=[]
    t_noise=[]
    con_showers=[]
    con_flashers=[]
    con_noise=[]

    for ev in range(len(sr_data[0])):
        if sr_data[3][ev]>cuts[1] and sr_data[3][ev]<cuts[0] and sr_data[4][ev]>cuts[3] and sr_data[4][ev]<cuts[2]:
          ch_showers.append(ev)
        elif sr_data[3][ev]>cuts[5] and sr_data[3][ev]<cuts[4] and sr_data[4][ev]>cuts[7] and sr_data[4][ev]<cuts[6]:
            ch_flashers.append(ev)
        else:
            ch_noise.append(ev)

        if sr_data[3][ev]>cuts[9] and sr_data[3][ev]<cuts[8] and sr_data[2][ev]>cuts[11] and sr_data[2][ev]<cuts[10]:
            t_showers.append(ev)
        elif sr_data[3][ev]>cuts[13] and sr_data[3][ev]<cuts[12] and sr_data[2][ev]>cuts[15] and sr_data[2][ev]<cuts[14]:
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

    
    charge_showers=np.zeros((2,len(ch_showers)))
    for ind, ev in enumerate(ch_showers):
        charge_showers[0][ind]=ev
        charge_showers[1][ind]=sr_data[5][ev]
    
    charge_flashers=np.zeros((2,len(ch_flashers)))
    for ind, ev in enumerate(ch_flashers):
        charge_flashers[0][ind]=ev
        charge_flashers[1][ind]=sr_data[5][ev]

    charge_noise=np.zeros((2,len(ch_noise)))
    for ind, ev in enumerate(ch_noise):
        charge_noise[0][ind]=ev
        charge_noise[1][ind]=sr_data[5][ev]

    time_showers=np.zeros((2,len(t_showers)))
    for ind, ev in enumerate(t_showers):
        time_showers[0][ind]=ev
        time_showers[1][ind]=sr_data[5][ev]

    time_flashers=np.zeros((2,len(t_flashers)))
    for ind, ev in enumerate(t_flashers):
        time_flashers[0][ind]=ev
        time_flashers[1][ind]=sr_data[5][ev]

    time_noise=np.zeros((2,len(t_noise)))
    for ind, ev in enumerate(t_noise):
        time_noise[0][ind]=ev
        time_noise[1][ind]=sr_data[5][ev]

    conf_showers=np.zeros((2,len(con_showers)))
    for ind, ev in enumerate(con_showers):
        conf_showers[0][ind]=ev
        conf_showers[1][ind]=sr_data[5][ev]

    conf_flashers=np.zeros((2,len(con_flashers)))
    for ind, ev in enumerate(con_flashers):
        conf_flashers[0][ind]=ev
        conf_flashers[1][ind]=sr_data[5][ev]

    conf_noise=np.zeros((2,len(con_noise)))
    for ind, ev in enumerate(con_noise):
        conf_noise[0][ind]=ev
        conf_noise[1][ind]=sr_data[5][ev]
        
    if list==True:
       print('Showers:',len(con_showers),'\nFlahsers:', len(con_flashers), '\nNoise:', len(con_noise), '\nCharge Showers:',len(ch_showers),'\nCharge Flashers:', len(ch_flashers),'\nCharge Noise:',len(ch_noise),"\nTime Showers:",len(t_showers),'\nTime Noise',len(t_noise))
       
    return conf_showers, conf_flashers, conf_noise, charge_showers, charge_flashers, charge_noise, time_showers, time_flashers, time_noise
    
#sorts the data into 9 lists, should be used to create the sorted_data object which has 9 sections with 2 indexes each

def event_rate(sorted_data, sr_data, run_id, sr_number, save_location, mod=1, test=False):
    modifier=mod
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.hist(sr_data[5], weights = [modifier for _ in range(len(sr_data[5]))], bins = np.arange(sr_data[5][0], sr_data[5][-1], 1E9/modifier), log=True, histtype = 'step', label = 'All') 
    ax.hist(sorted_data[0][1], weights = [modifier for _ in range(len(sorted_data[0][1]))], bins = np.arange(sorted_data[0][1][0], sorted_data[0][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'Showers') 
    ax.hist(sorted_data[1][1], weights = [modifier for _ in range(len(sorted_data[1][1]))], bins = np.arange(sorted_data[1][1][0], sorted_data[1][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'Flashers') 
    ax.hist(sorted_data[2][1], weights = [modifier for _ in range(len(sorted_data[2][1]))], bins = np.arange(sorted_data[2][1][0], sorted_data[2][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'Other') 
    
    if test==True:
       ax.hist(sorted_data[3][1], weights = [modifier for _ in range(len(sorted_data[3][1]))], bins = np.arange(sorted_data[3][1][0], sorted_data[3][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'charge showers') 
       ax.hist(sorted_data[6][1], weights = [modifier for _ in range(len(sorted_data[6][1]))], bins = np.arange(sorted_data[6][1][0], sorted_data[6][1][-1], 1E9/modifier), log=True, histtype = 'step', label = 'time showers') 
   
    ax.legend(loc='upper left')
    ax.set_title(f"Event Rates Run {run_id}, sr{sr_number}")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Rate [Hz]")
    fig.savefig(save_location)
    print("summary plot B has actually been generated")
    plt.show()

#makes histogram of rates with or without some test lines that help with refining box cuts  
def event_rate_summary(run_id, sr_number, r0_location, tcal_location, r1_location, save_location, test=False, resolution=1):

    sr_id=sr_number
    r0_file=r0_location+'run'+str(run_id)+'_subrun'+str(sr_id)+'_r0.tio'

    tcal_file=tcal_location

    r1_file=r1_location+'run'+str(run_id)+'_subrun'+str(sr_id)+'.r1'

    reader=get_reader(r0_file, tcal_file, r1_file)

    sr_data=collect_stats(reader) 


    cuts=get_cuts()
    
    get_hists(sr_data, cuts, display=test, regions= test, flashers= test, boxes= test)

    sorted_data=sort_data(sr_data, cuts, list=test)

    event_rate(sorted_data, sr_data, run_id, sr_number, save_location, test=test, mod=resolution)

    #Produces the whole nice graph thing

#OTHER METRICS FUNCTION

def physical_summary(run_id, sr_number, metrics_location, save_location, modules=22):
    run=run_id
    subruns=sr_number+1
    
    fpmTemp_list=[]
    feeTemp_list=[]
    hv_list=[]
    current_list=[]

    for sr in range(subruns):

      fpmTemp_data=np.load(f'{metrics_location}temperatures_FPMs_run{run}_subrun{sr}.npy')
      fpmTemp_list.append(fpmTemp_data)

      feeTemp_data=np.load(f'{metrics_location}temperatures_FEEs_run{run}_subrun{sr}.npy')
      feeTemp_list.append(feeTemp_data)

      hv_data=np.load(f'{metrics_location}HV_FPMs_run{run}_subrun{sr}.npy')
      hv_list.append(hv_data)

      current_data=np.load(f'{metrics_location}Current_FEEs_run{run}_subrun{sr}.npy')
      current_list.append(current_data)

    fpmTemps=np.zeros((modules*4,3,len(fpmTemp_list[:])))
    feeTemps=np.zeros((modules*2,3,len(feeTemp_list[:])))
    hv=np.zeros((modules,3,len(hv_list[:])))
    current=np.zeros((modules,3,len(current_list[:])))

    for sr in range(subruns):

        for quad in range(modules*4):
           fpmTemps[quad][0][sr]=sr
           fpmTemps[quad][1][sr]=fpmTemp_list[sr][quad]
           fpmTemps[quad][2][sr]=quad//4

        for board in range(modules*2):
           feeTemps[board][0][sr]=sr
           feeTemps[board][1][sr]=feeTemp_list[sr][board]
           feeTemps[board][2][sr]=board//2

        for mod in range(modules):
           hv[mod][0][sr]=sr
           hv[mod][1][sr]=hv_list[sr][mod]
           hv[mod][2][sr]=mod
           current[mod][0][sr]=sr
           current[mod][1][sr]=current_list[sr][mod]
           current[mod][2][sr]=mod

    fig, (ax1,ax2,ax3,ax4)=plt.subplots(4,1)

    for quad in range(modules*4):
       ax1.plot(fpmTemps[quad][0], fpmTemps[quad][1])
    ax1.set_ylabel("FPM Temp (C)")

    for board in range(modules*2):
       ax2.plot(feeTemps[board][0], feeTemps[board][1])
    ax2.set_ylabel("FEE Temp (C)")

    for mod in range(modules):
       ax3.plot(hv[mod][0],hv[mod][1])
       ax4.plot(current[mod][0],current[mod][1])
    ax3.set_ylabel("HV (V)")
    ax4.set_ylabel("Current (A)")
    ax4.set_xlabel("Subrun ID")
    fig.tight_layout()
    ax1.set_title(f"Run {run_id} sr 0-{sr_number} physical metrics")
    fig.savefig(save_location)
    print("summary plot A has actually been generated")
    plt.show()
    
#CONDENSED FUNCTION 

def sr_summary(run_id, sr_number, metrics_location, r0_location, tcal_location, r1_location, ev_save_location, phys_save_location, test=False, resolution=1, modules=22):

    event_rate_summary(run_id, sr_number, r0_location, tcal_location, r1_location, ev_save_location, test=False, resolution=resolution)
    print("Summary plot B has been generated")

    physical_summary(run_id, sr_number, metrics_location, phys_save_location, modules=modules)
    print("Summary plot A has been generated")

    print(f"Run {run_id} sr{sr_number} summary")

#NEWEST FILE DETECTOR function

def get_new_sr(run_base=400195, subrun_base=0):

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

active=True
last_sr=15
while active==True: #This is the important cell that manages everything
   
    current_sr=get_new_sr()
    print(f"Latest Subrun: Run {current_sr[0]} sr{current_sr[1]}")

    if os.path.exists(f"{physical_metrics_location}Current_FEEs_run{current_sr[0]}_subrun{current_sr[1]}.npy")==True:
       tim_n = time.time()
       sr_summary(current_sr[0], current_sr[1], physical_metrics_location, r0_file_location, pedestal_path, new_r1_file_location, summary_plots_location_1, summary_plots_location_2)
       print(f'Summary for Subrun {current_sr[1]} took {time.time()-tim_n}s')
    else:
        print("This file doesn't exist yet")
        time.sleep(5)
    next_sr=get_new_sr()
    while current_sr==next_sr:
       next_sr=get_new_sr()

       if current_sr[1]==last_sr: #This line just stops this from going forever, comment out when not working with simulator
           active=False
           break

