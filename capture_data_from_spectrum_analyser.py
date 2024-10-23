#!/usr/bin/env python/
# -*- coding: utf-8 -*-
"""A script to use a spectrum analyser for cold sky hot load measurements.

This script uses the Rohde & Schwarz FPH spectrum analyser to carry out
hot load cold sky measurements and plot the equivalent noise temperature, gain,
flatness and ripple.

Script version: V1 (this was not tracked before)
Originally created: 15 September 2020
Original authors: S. Malan, W. Newton, M. Venter
Last update date: 19/07/2024 by W. Newton
Last update reason: Update script for Ku band testing

"""
# %% Import functions that do the work
import numpy as np
import pyvisa
# import time
from numpy import double
import os
from datetime import datetime
import matplotlib.pyplot as plt
import scipy
# %% Global constants and variable definitions
# -----------------------------------------------------------------------------
k = 1.38e-23  # Boltzman's constant
f_center = 3200  # Measurement center frequency in MHz
f_bw = 6000  # Measurement frequency BW in MHz
f_start = (f_center - f_bw/2)  # Measurement start frequency in MHz
f_stop = (f_center + f_bw/2)  # Measurement stop frequency in MHz
point_spacing = 1  # Required space between points in MHz
f_points = f_bw  # Number of measurement points

# # Generate the frequency array
# freq = np.arange(f_points+1)*point_spacing+f_start

# Spectrum Analyser settings
# -----------------------------------------------------------------------------
rbw = 3  # RBW of spectrum analyser in MHz
vbw = 0.001  # VBW of spectrum analyser in MHz
RefLev = -8  # Reference level of spectrum analyser in dBm
Atten = 5  # Spectrum analyser attenuation in dB. A minimum of 5 dB
sweeps = 5  # Number of sweeps to capture for each measurement

# Test settings and notes
# -----------------------------------------------------------------------------
band = 'H'  # directory that the data will be saved to
meas = 'Ku-band SN001 initial sky test 118_3 MHz'  # file name. Specify a meaninful name

#%%

initial_file_name = band+'/'+band+' '+meas

'''suggested naming convention
teff = np.load(band+'/Teff_'+band+'_'+meas+'.npy')
teff_5k = np.load(band+'/Teff_'+band+'_'+meas+'_5k_nd.npy')
teff_20k = np.load(band+'/Teff_'+band+'_'+meas+'_20k_nd.npy')
'''

# Make a note of all these settings during testing. Will save to the log file
test_location = 'Receiver Lab Liesbeek House'
device_under_test = 'Ku-band test receiver SN001'
# On the Ghana dish this was used to specify whether the hot load was
# above the feedhorn or above the beam wave guide.
hot_load_location = 'On the feedhorn'
# From https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/277315585/MeerKAT+specifications#System-temperature
#  Tsky=  2.725 + 1.6(ν/GHz)-2.75
t_cold_K = 2.725 + 1.6*(f_center/1000)**(-2.75)  # Cold reference (cold sky)
t_hot_C = 25.0  # Measured hot load temperature in degree C (ambient load)
t_hot_K = t_hot_C + 274.15
sky_conditions = 'clear'  # it is good practice to record this
wind = 'none'
comment = 'Initial testing carried out after repair of the power circuit and REF clock generator Ref set to 118.3 MHz'  # Any additional comments. Perhaps, the test intention

# %% Functions
# -----------------------------------------------------------------------------


def initialise_log_file(file_name):
    """Initialise a new log file, iterate the file name if one already exists.

    Args:
        log_file_name (string): The path and filename.

    Returns
    -------
        new_log_file_name (string): The path and filename
        append (bool): Whether or not the file already existed
        i (int): The iteration number to append to the filename

    Note:
        The function assumes that the global variables are available:
        test_location,
        device_under_test,
        hot_load_location,
        t_cold_K,
        t_hot_C,
        t_hot_K,
        sky_conditions,
        wind,
        comment

    Example:
        log_file, file_exists, file_append_nr =
        initialise_log_file(band+'/'+band+'_'+meas)
    """
    # log_file_name = file_name + ' log'
    new_file_name = file_name  # initially they will be the same
    # Only create a new log file if it does not exist
    # check to see if the log file already exists
    i = 1
    while os.path.exists(new_file_name+' log' + '.txt'):
        new_file_name = file_name+' '+str(i)
        print('file exists trying this name:\n %s' %
              (new_file_name+' log' + '.txt'))
        i += 1
        # append = True

    print('new log file created %s' % new_file_name+' log' + '.txt')
    # open the log file with write privelages
    with open(new_file_name+' log' + '.txt', 'w') as log_file:
        log_file.write('Start date and time:' +
                       (datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        log_file.write('\nTest location: %s\n' % test_location)
        log_file.write('Device under test: %s\n' % device_under_test)
        log_file.write('Hot load location: %s\n' % hot_load_location)
        log_file.write('Cold reference: %.2f (K)\n' % t_cold_K)
        log_file.write('Hot reference: %.2f (K) %.2f (C)\n' % (t_hot_K,
                                                               t_hot_C))
        log_file.write('Sky conditions: %s\n' % (sky_conditions))
        log_file.write('Wind: %s\n' % wind)
        log_file.write('Comments: %s\n' % comment)
        log_file.write('%s \t %s \n' % (
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'Logging started'))
    return new_file_name
# %%


def write_to_log_file(local_file_name, contents):
    """Append data to an existing log file.

    Args:
        file_name (string): The path and filename.
        contents (string): The data to append to the file.

    Returns
    -------
        Nothing

    Note:
        The function assumes that the global variables are available:


    Example:
        write_to_log_file(file_name, 'hot measurement started')
    """
    # open the log file with append privelages
    with open(local_file_name+' log' + '.txt', 'a') as log_file:
        log_file.write('%s \t %s \n' % (
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"), contents))

    return


def set_sa_bw(rbw, vbw):
    """Set the resolution and video bandwidths of a spectrum analyzer (SA).

    This function configures the resolution bandwidth (RBW) and video bandwidth
    (VBW) of a spectrum analyzer (SA). It sends commands to set these
    bandwidths, verifies the settings, and prints the current status of the
    bandwidth settings.

    Args:
        rbw (int): The desired resolution bandwidth (RBW) in MHz.
        vbw (int): The desired video bandwidth (VBW) in MHz.

    Returns
    -------
        None

    Sends commands to the spectrum analyzer (SA) to set the RBW and VBW to the
    specified values. It then queries the SA to verify the current settings and
    whether the bandwidths are set to AUTO mode. Finally, it prints the RBW and
    VBW settings in KHz, along with the AUTO mode status.

    Note:
        The function assumes that the `sa` object, representing the spectrum
        analyzer, is available in the scope where this function is called.
        The global variable working_file_name should be available

    Example:
        To set the resolution bandwidth to 3 MHz and video bandwidth to 1 MHz
        set_sa_bw(3, 1)
    """
    # Send the command to set the resolution BW to rbw in MHz
    sa.write('BAND %s MHz' % rbw)
    # Check what the RBW reported by the SA is
    rbw_set = float(sa.query('BAND?'))
    # Check if RBW is set to AUTO mode
    rbw_auto_set = int(sa.query('BAND:AUTO?'))
    # Send the command to set the video BW to vbw in MHz
    sa.write('BAND:VID %s MHz' % vbw)
    # Check what the VBW reported by the SA is
    vbw_set = float(sa.query('BAND:VID?'))
    # Check if VBW is set to AUTO mode
    vbw_auto_set = int(sa.query('BAND:VID:AUTO?'))

    print('SA RBW set to AUTO %i, RBW = %0.2f Hz' % (rbw_auto_set,
                                                      rbw_set * 1e3))
    print('SA VBW set to AUTO %i, VBW = %0.2f Hz' % (vbw_auto_set,
                                                      vbw_set * 1e3))
    # save the set values to the log file using the global working_file_name
    write_to_log_file(working_file_name,
                      'SA RBW set to AUTO %i, RBW = %0.2f Hz' % (
                          rbw_auto_set, rbw_set * 1e3))
    write_to_log_file(working_file_name,
                      'SA VBW set to AUTO %i, VBW = %0.2f Hz' % (
                          vbw_auto_set, vbw_set * 1e3))
    return


def set_sa_detect(det_mode, sweeps):
    """Set the detector mode and number of sweeps for a spectrum analyzer (SA).

    This function configures the detector mode and the number of sweeps for a
    spectrum analyzer (SA). It sends commands to set these parameters and
    verifies the current settings.

    Args:
        det_mode (str): The desired detector mode. Options are:
                            'APE' (Autopeak),
                            'POS',
                            'NEG',
                            'SAMP',
                            'RMS',
                            'AVER' (Average),
                            'QPE' (Quasipeak).
        sweeps (int): The number of sweeps to average.
                        Will only work if average is enabled
        #TODO: trace_mode (str) : Sets the desired trace mode. Options are:
                            'WRIT' (Clear/Write),
                            'MAXH',
                            'AVER',
                            'VIEW'

    Returns
    -------
        None

    Sends commands to the spectrum analyzer (SA) to set the detector mode and
    the number of sweeps. It then queries the SA to verify the current settings
    and prints the detector mode and trace mode.

    Note:
        The function assumes that the `sa` object, representing the spectrum
        analyzer, is available in the scope where this function is called.
        The global variable working_file_name should be available

    Example:
        set_sa_detect('RMS', sweeps)
    """
    sa.write('DET %s' % det_mode.upper())
    # set the amount of sweeps to average
    # TODO self.sa.write("AVER:COUN %i" %sweeps)
    # switches on the calculation of average.
    # TODO self.sa.write("AVER:STAT ON")
    # check the detector mode set and print
    print('detector: %s' % sa.query('DET?'))
    # check the trace mode set and print
    print('trace mode set to %s ' % sa.query('DISP:WIND:TRAC:MODE?'))
    # save the set values to the log file using the global working_file_name
    write_to_log_file(working_file_name,
                      'SA trace mode set to %s' % sa.query(
                          'DISP:WIND:TRAC:MODE?'))
    write_to_log_file(working_file_name,
                      'detector: %s' % sa.query('DET?'))
    return


def set_sa_amplitude(ref_level_dBm, att_level_dB):
    """Set the ref level and attenuation level of the spectrum analyzer (SA).

    Args:
        ref_level_dBm (int): The desired reference level in dBm.
        att_level_dB (int): The desired input attenuator value in dB.

    Returns
    -------
        None

    Note:
        The function assumes that the `sa` object, representing the spectrum
        analyzer, is available in the scope where this function is called.
        The global variable working_file_name should be available

    Example:
        To set the reference level to -20 dBm and the input attenuator to 5 dB
        set_sa_amplitude(-20, 5)
    """
    # Send command to switch to manual attenuation
    sa.write('INP:ATT:AUTO OFF')
    # Send command to set reference level
    sa.write('DISP:WIND:TRAC:Y:RLEV %sdBm' % ref_level_dBm)
    # Send command to set the input attenuator
    sa.write('INP:ATT %s dB' % att_level_dB)
    # Check the set value for reference level
    print('SA amplitude reference level set to REF %0.2f dBm' % (
        double(sa.query('DISP:WIND:TRAC:Y:RLEV?'))))
    # Check the set value for input attenuator
    print('SA input attenuator set to %s dB' % (
        double(sa.query('INP:ATT?'))))
    # Write the values to log file
    write_to_log_file(working_file_name,
                      'SA amplitude reference level set to REF %0.2f dBm' % (
                          double(sa.query('DISP:WIND:TRAC:Y:RLEV?'))))
    write_to_log_file(working_file_name,
                      'SA input attenuator set to %s dB' % (
                          double(sa.query('INP:ATT?'))))
    return


def sa_getsweepdata():
    """Initiate the sweep, check for operation complete and return the trace.

    Args:
        none

    Returns
    -------
        sweepdata (list): The sweep data in dBm

    Note:
        The function assumes that the `sa` object, representing the spectrum
        analyzer, is available in the scope where this function is called.

    Example:
        To carry out the sweep and collect the data
        sweep = sa_getsweepdata()
    """
    # switch to single sweep mode on display 1
    sa.write('INIT1:CONT OFF')
    # Initiate the sweep and wait for the operation complete to return a 1
    sa.query("INIT1;*OPC?")

    return list(eval(sa.query('TRACE1? TRACE1')))


def sa_save_hot_trace(local_file_name,
                      sweeps_local,
                      f_points_local,
                      freq_local):
    """Capture the measurement sweeps with the hot load attached.

    Args:
        local_file_name (): The subdirectory and file name to save the data
        sweeps_local (int): Number of trace sweeps to capture
        f_points_local (int): Number of frequency points in the trace
        freq_local (array of float 64): The array of frequency points
                                        corresponding to each trace data point

    Returns
    -------
        Nothing

    Note:
        The f_points_local number must be equal to the length of the
        'freq_local' array.
        The function assumes that the `sa` object, representing the spectrum
        analyzer, is available in the scope where this function is called.

    Example:
        To carry out the hot load measurement and save the file to the
        current working file name with 20 sweeps, 100 sweep points with
        a frequency array 'freq':
        sa_save_hot_trace(working_file_name, 20, 100, freq)
    """
    # array for storing trace data
    trace_dbm = np.zeros((sweeps_local+1, f_points_local), dtype=float)
    # array for storing phot data
    p_meas_w = np.zeros((sweeps_local+1, f_points_local), dtype=float)
    # Store the frequency points as row 0 in the dataTrace array
    trace_dbm[0, :] = np.array(freq_local, dtype=float)
    # Store the frequency points as row 0 in the power array
    p_meas_w[0, :] = np.array(freq_local, dtype=float)

    # Measure Pout_hot
    keyIn = input('Connect hot load. Ready? (y/n):')
    if keyIn == 'y':
        i = 1  # Start saving the trace data only from the second row
        for i in range(1, sweeps_local+1):
            # Fetch the measurement data
            temp = sa_getsweepdata()
            # Get trace data and change data into a numpy array (dBm)
            trace_dbm[i, :] = np.array(temp, dtype=float)
            # Convert the coptured trace data to linear power (W)
            p_meas_w[i, :] = (10**(trace_dbm[i, :]/10)*1e-3)
            print("Tref_hot measurement # %i of %i" % (i, sweeps_local))
        # Save the linear measured power (W)
        np.save(local_file_name + ' hot W', p_meas_w)
        # Save the log measured power (dBm)
        np.save(local_file_name + ' hot dBm', trace_dbm)
    return


def sa_save_cold_trace(local_file_name,
                       sweeps_local,
                       f_points_local,
                       freq_local):
    """Capture the measurement sweeps with the cold load attached.

    Args:
        local_file_name (): The subdirectory and file name to save the data
        sweeps_local (int): Number of trace sweeps to capture
        f_points_local (int): Number of frequency points in the trace
        freq_local (array of float 64): The array of frequency points
                                        corresponding to each trace data point

    Returns
    -------
        Nothing

    Note:
        The f_points_local number must be equal to the length of the
        'freq_local' array.
        The function assumes that the `sa` object, representing the spectrum
        analyzer, is available in the scope where this function is called.

    Example:
        To carry out the cold load measurement and save the file to the
        current working file name with 20 sweeps, 100 sweep points with
        a frequency array 'freq':
        sa_save_hot_trace(working_file_name, 20, 100, freq)
    """
    # array for storing trace data
    trace_dbm = np.zeros((sweeps_local+1, f_points_local), dtype=float)
    # array for storing phot data
    p_meas_w = np.zeros((sweeps_local+1, f_points_local), dtype=float)
    # Store the frequency points as row 0 in the dataTrace array
    trace_dbm[0, :] = np.array(freq_local, dtype=float)
    # Store the frequency points as row 0 in the power array
    p_meas_w[0, :] = np.array(freq_local, dtype=float)

    # Measure Pout_cold
    keyIn = input('Connect cold load. Ready? (y/n):')
    if keyIn == 'y':
        i = 1  # Start saving the trace data only from the second row
        for i in range(1, sweeps_local+1):
            # Fetch the measurement data
            temp = sa_getsweepdata()
            # Get trace data and change data into a numpy array (dBm)
            trace_dbm[i, :] = np.array(temp, dtype=float)
            # Convert the coptured trace data to linear power (W)
            p_meas_w[i, :] = (10**(trace_dbm[i, :]/10)*1e-3)
            print("Tref_cold measurement # %i of %i" % (i, sweeps_local))
        # Save the linear measured power (W)
        np.save(local_file_name + ' cold W', p_meas_w)
        # Save the log measured power (dBm)
        np.save(local_file_name + ' cold dBm', trace_dbm)
    return


# %% Main program
# -----------------------------------------------------------------------------

# Initialise the log file, the working file name is returned
working_file_name = initialise_log_file(initial_file_name)

# Initialise pyvisa
rm = pyvisa.ResourceManager()
# Open the spectrum analyser - For the Spectrum Rider FPH, a manual IP needs to be set
sa = rm.open_resource('TCPIP::10.28.1.205::INSTR')  # The VXI-11 protocol is not socket
sa.read_termination = '\n'  # Define termination characters
sa.write_termination = '\n'
sa.timeout = 50000
# Test the connection to the sa by querying the *IDN? SCPI command
print('Connected to: ' + sa.query("*IDN?"))

# %% Initialise the Spectrum analyser
sa.write('*CLS')  # clear
sa.write('*RST')  # reset
sa.write('INST:SEL SAN')  # ?
sa.write('SYST:DISP:UPD ON')  # Update the display during measurement


# Set the start freq, stop frequency and number of points
sa.write('FREQ:STAR %s MHz' % f_start)  # set the start frequency
sa.write('FREQ:STOP %s MHz' % f_stop)  # set the stop frequency
sa.write('SWE:POIN %s' % f_points)  # set the number of sweep points

# Read back the set data in the spectrum analyser
# Read the start freq
f_start_set = int(double(sa.query('FREQ:STAR?')))
# Read back the stop freq
f_stop_set = int(double(sa.query('FREQ:STOP?')))
# Read back the number of sweep points
f_points_set = int(double(sa.query('SWE:POIN?')))

# Set the SA Resolution and video bandwidth
set_sa_bw(rbw, vbw)
# Set SA detector Detector Mode to RMS
# Sweep mode to Average,
# Amount of Sweeps to average
set_sa_detect('RMS', sweeps)
# Set SA amplitude reference level and input attenuator setting
set_sa_amplitude(RefLev, Atten)

sa.write('INP:GAIN:STAT ON') # Activates the preamplifier

#%%
# Generate the frequency array again based on what the SA is set tp
freq = np.linspace(f_start_set,f_stop_set,f_points_set)

# %% Carry out measurement
# Note the time that the hot load measurement started in the log file
write_to_log_file(working_file_name,
                  'Starting hot load measurement')
# Carry out the hot load measurement
sa_save_hot_trace(working_file_name,
                  sweeps,
                  f_points_set,
                  freq)
# Note the time the hot load measurement ended and cold load measurement begins
write_to_log_file(working_file_name,
                  'Starting cold load measurement')
# Carry out the cold load measurement
sa_save_cold_trace(working_file_name,
                   sweeps,
                   f_points_set,
                   freq)

# %% Time to plot the result

# Load the save trace data from file

# Set the initial file = to the working file name. This is only needed for analysis after the measurements
# working_file_name = initial_file_name


# Measured output noise power with the hot load attached (W)
p_meas_hot = np.load(working_file_name + ' hot W.npy')
# Measured output noise power with the cold load attached (W)
p_meas_cold = np.load(working_file_name + ' cold W.npy')






# %% Average the data

# Initialise the arrays
p_out_hot_ave = np.zeros((2, len(p_meas_hot[1])), dtype=float)
t_out_hot_ave = np.zeros((2, len(p_meas_hot[1])), dtype=float)
p_out_hot_dBm = np.zeros((2, len(p_meas_hot[1])), dtype=float)
p_out_cold_dBm = np.zeros((2, len(p_meas_hot[1])), dtype=float)

# Copy the frequency points from the loaded array to the average array
p_out_hot_ave[0, :] = p_meas_hot[0, :]
t_out_hot_ave[0, :] = p_meas_hot[0, :]
p_out_hot_dBm[0, :] = p_meas_hot[0, :]
p_out_cold_dBm[0, :] = p_meas_hot[0, :]
p_out_cold_ave = np.zeros((2, len(p_meas_hot[1])), dtype=float)
t_out_cold_ave = np.zeros((2, len(p_meas_hot[1])), dtype=float)
# Copy the frequency points from the loaded array to the average array
p_out_cold_ave[0, :] = p_meas_cold[0, :]
t_out_cold_ave[0, :] = p_meas_hot[0, :]

Y_factor = np.zeros((2, len(p_meas_hot[1])), dtype=float)
# Copy the frequency points from the loaded array to the Y-factor array
Y_factor[0, :] = p_meas_hot[0, :]
T_eff = np.zeros((2, len(p_meas_hot[1])), dtype=float)
T_eff[0, :] = p_meas_hot[0, :]
Gain_dut = np.zeros((2, len(p_meas_hot[1])), dtype=float)
Gain_dut[0, :] = p_meas_hot[0, :]
# Array for the equivalent noise temperature of the device under test (DUT)
Te_dut = np.zeros(len(p_meas_hot[1]), dtype=float)

# Calculate the averages
i = 0
for i in range(0, len(p_meas_hot[1])):
    # Calculate average measured hot power for each frequency bin
    # Exclude the first row, that is the actual frequency
    p_out_hot_ave[1, i] = np.average(p_meas_hot[1:, i])
    # Calculate average measured cold power for each frequency bin
    # Exclude the first row, that is the actual frequency
    p_out_cold_ave[1, i] = np.average(p_meas_cold[1:, i])

# Calculate measured output temperature, convert rbw from MHz to Hz
t_out_hot_ave[1, :] = ((p_out_hot_ave[1, :]))/(k*rbw*1E6)
# Calculate measured output temperature
t_out_cold_ave[1, :] = ((p_out_cold_ave[1, :]))/(k*rbw*1E6)

# %%
# Method as described in (EA-MK-000-DREP-09_2)
i = 0
for i in range(0, len(t_out_cold_ave[1])):
    # Calculate the Y-factor for each frequency point
    Y_factor[1, i] = p_out_hot_ave[1, i]/p_out_cold_ave[1, i]
    # Calculate the effective measured noise temperature for each point
    T_eff[1, i] = (t_hot_K-t_cold_K*Y_factor[1, i])/(Y_factor[1, i]-1)
    # Calculate the DUT gain in dB
    Gain_dut[1, i] = 10*np.log10(
        (t_out_hot_ave[1, i]-t_out_cold_ave[1, i])/(t_hot_K-t_cold_K))

# Convert power from watts to dBm to power
p_out_hot_dBm[1, :] = 10*np.log10(p_out_hot_ave[1, :]/1e-3)
p_out_cold_dBm[1, :] = 10*np.log10(p_out_cold_ave[1, :]/1e-3)

# %% Plot Data
# -----------------------------------------------------------------------------

# varables used to trim data down
data_s = 82
data_f = 174  # 2501

# Declare the Plotting functions


def plot_gain():
    ''' '''

    plt.figure(1)
    plt.clf()

    z = np.polyfit(Gain_dut[0, data_s: data_f],
                   Gain_dut[1, data_s: data_f],
                   1)
    p = np.poly1d(z)

    plt.plot(freq[data_s:data_f]/1e6,
             p(freq[data_s:data_f]),
             color='orange',
             alpha=0.8,
             label='slope = %2.2f dB' % (400*(p.c[0])))

    plt.plot(Gain_dut[0, data_s: data_f]/1e6,
             Gain_dut[1, data_s: data_f],
             linewidth=1,
             color='r',
             label ='ripple = %2.2f dB' % (
                 (np.max(p(freq[data_s:data_f])-(Gain_dut[1, data_s:data_f]))+np.max((Gain_dut[1, data_s:data_f])-p(freq[data_s:data_f])))))




    print(np.mean(Gain_dut[1, data_s:data_f]))
    print(np.std(Gain_dut[1, data_s:data_f]))

    plt.legend(loc='lower center')
    plt.title(working_file_name+" gain slope and gain ripple")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Gain(dB)")
    plt.grid(which='both',color='black', linestyle='-', linewidth=0.5)
    # plt.ylim((20,60))
    # plt.xlim((f_start, f_stop))
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)
    fig.savefig(working_file_name+' gain.png', dpi=200)
    return


# %%

def plot_ripple():
    plt.figure(2)
    plt.clf()    
        
    z = np.polyfit(Gain_dut[0, data_s:data_f],
                   Gain_dut[1, data_s:data_f],
                   1)
    p = np.poly1d(z)

    plt.plot(Gain_dut[0, data_s:data_f],
             p(Gain_dut[0, data_s:data_f])-(Gain_dut[1, data_s:data_f]),
             color='orange',
             alpha=0.8,
             label=' peak to peak gain ripple = %2.2f dB' % (np.max(p(Gain_dut[0, data_s:data_f])-(Gain_dut[1, data_s:data_f]))+np.max((Gain_dut[1, data_s:data_f])-p(Gain_dut[0, data_s:data_f]))))
    plt.legend(loc='best')
    plt.title(working_file_name+" gain ripple normalised to gain slope")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Gain ripple (dB)")
    plt.grid(b='on',
             which='both',
             color='black',
             linestyle='-',
             linewidth=0.5)
#    plt.ylim((50,65))
    # plt.ylim((-2, 2))
#    plt.ylim((00,15))
#    plt.xlim((700,832))
    # plt.xlim((f_start, f_stop))
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)
    fig.savefig(working_file_name + ' ripple.png', dpi=200)
    return

#%% 

def plot_T_eff():

    plt.figure(3)
    plt.clf()
    plt.plot(T_eff[0, data_s:data_f]/1e6,
             T_eff[1, data_s:data_f],
             linewidth=1,
             color='b',
             label='Measured $T_e$')
    plt.plot([900,1670],[90,90], linewidth=1,color='r', label='Specification = <90 K')


    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Equivalent Noise Temperature (K)")
    plt.title(working_file_name+" equivalent noise temperature")
    plt.legend(loc="best")
    # plt.ylim((0, 400))
  #  plt.xlim((f_start, f_stop))
    plt.grid(which='both',
             color='black',
             linestyle='-',
             linewidth=0.5,
             alpha=0.5)
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)
    fig.savefig(working_file_name + ' Te.png', dpi=300)
    return

# %%


def plot_power():

    plt.figure(4)
    plt.clf()
    plt.plot(p_out_hot_dBm[0, data_s:data_f],
             p_out_hot_dBm[1, data_s:data_f],
             linewidth=1,
             color='r',
             label='Pout (hot load)')
    plt.plot(p_out_cold_dBm[0, data_s:data_f],
             p_out_cold_dBm[1, data_s:data_f],
             linewidth=1,
             color='b',
             label='Pout (cold load)')
    plt.legend(loc='upper right')
    plt.title(working_file_name+" measured power")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power(dBm)")
    plt.grid(which='both',
             color='black',
             linestyle='-',
             linewidth=0.5)
#    plt.ylim((-120,-20))
    # plt.xlim((f_start, f_stop))
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)
    fig.savefig(working_file_name + ' powers.png', dpi=200)
    return


# %%

plot_power()
plot_T_eff()
plot_gain()
