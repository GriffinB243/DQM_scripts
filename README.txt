General info: DQM_script and DQM_config are the most important, to learn more about DQM script jump to the bottom and read that loop

read alongside DQM_config for explanation of options

r0_file_location: full path of r0 files

pedestal_path: full path of chosen pedestal

r1_file_location: full path for where created r1 files should be saved

current_file: full path of the current .npys

hv_file: full path of the hv .npys

FEE_temp_file: full path of the fee temp .npys

FPM_temp_file: full path of the fpm temp .npys

modules: number of modules currently installed in the camera

monitoring: 1=true/0=false, will the script do anything

live: 1=true/0=false, will change how initial run is taken to interpret it as latest run and start start monitoring waiting for intial run +1

initial_run: run monitoring should start looking for at subrun 0. should theoretically work to start looking at a run from the start even if a few subruns are already done

final_run: end point, not currently particularly useful here. should stop running the script before looking at this run/ when this run starts.

runtime_test: 1=true/0=false, gets run time of different parts of the script and plots them to number of events to test script performance. Note this will only happen if the script reaches its final subrun, not if you stop it manually.

histograms_1d: 1=true/0=false, will 1d histograms be generated

histograms_2d: 1=true/0=false, will 2d histograms be generated

histograms_dt: 1=true/0=false, will dt histograms be generated

subrun_plots: 1=true/0=false, will every plot generated cumulatively have a counterpart just within the subrun

boxes: 1=true/0=false, will sorting lines be visible on plots, currently results in some annoying legends referring to nothing

shower_regions: 1=true/0=false, will plots be generated zoomed in on shower regions of relevant plots

flasher_regions: 1=true/0=false, will plots be generated zoomed in on flasher regions of relevant plots

tight_windows: 1=true/0=false, changes the window of shower zooms specifically to make them focus around the origin/weaker area rather than showing all showers. should be made obselete by log option at some point

logscale: plots 2d histograms again in log10 scale which provides more detail so it's nice

bins: number of bins on the sorting histograms

resolution: how many bins on the event rate histogram per second. does have to be a float

time_step: value to convert between ns and desired time scale, has to be an int

errors: 1=true/0=false, will error bars be plotted on the event rate histogram. they're poisson errors for now

fontsize: fontsize on all text

charge_mean_flasher_min: minimum ADC*ns count to for an event to be a flasher

shower_intercept: shower cut is diagonal, simple -x + b. set intercept here, it has to be an int

envelopes: 1=true/0=false, extra lines and an alert on the environmental metric plots for unwanted values. they have to be floats

FEE_temp_high: value in C to be alerted if an FEE temperature passes above, these can be floats

FEE_temp_low: value in C to be alerted if an FEE temperature passes below

FPM_temp_high: value in C to be alerted if an FPM temperature passes above

FPM_temp_low: value in C to be alerted if an FPM temperature passes below

current_high: value in A to be alerted if a current value passes above

current_low: value in A to be alerted if a current value passes below

hv_high: value in V to be alerted if an HV value passes above

hv_low: value in V to be alerted if an HV value passes below

display_plots_path: path to a folder where display plots are saved. these plots will be overwritten as the script runs and should stay up to date to the last run

plots_save_path: path to a folder where each plot is saved. these plots will be the display plots for the run with every subrun the script has looked at so far, which will also be overwritten as the script runs through subruns but they should stay the same once the run changes. This folder also has the subrun plots which are all the display plots but made with just that subrun's data each subrun. there are a lot of them. Cumulative plots will either have only the run number in their names while subrun plots will specify run number and subrun number. 