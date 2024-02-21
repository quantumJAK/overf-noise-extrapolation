import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
# import estimation_method from parent directory
import sys
sys.path.append('../')

import estimation_methods as em
import noise_models as nm



def bimodal_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    '''
    Define a bimoal Gaussian function
    '''
    return (A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

def get_binary_signal(ID_dBz, path, file=None, number_of_measurements=None, leftind=0, rightind =10000):
    '''
    Get the binary signal matrix from the raw signal matrix
    Method  by Fabrizzio
    ----------------
    Parameters:
    ID_dBz: int
        ID of the experiment
    path: str
        Path to the data
    file: str
        Name of the file to be loaded
    number_of_measurements: int
        Number of measurements to be used
    leftind: int
        Left index of the data to be used
    rightind: int
        Right index of the data to be used
    Returns:
    binary_signal_matrix: np.array
        Binary signal matrix
      '''

    if ID_dBz is not None:
    ## fitting singlet states
        ID_raw_signal = ID_dBz-2
        raw_signal_matrix = np.load(path+f"signal_raw_{ID_raw_signal}.npy")[:,leftind:rightind]
    else:
        raw_signal_matrix = np.load(file)[:,leftind:rightind]
    # Flatten the raw_signal matrix
    flattened_signal = raw_signal_matrix.flatten()

    # Create a histogram of the flattened signal
    hist, bin_edges = np.histogram(flattened_signal, bins=50, density=False)
    
    #Control plot
    #plt.figure()
    
    #plt.hist(flattened_signal, bins=50)


    # New method of guessing initial parameters by JAK
    mmax = flattened_signal.max() #max of the flattened signal
    mmin = flattened_signal.min() #min of the flattened signal
    mmax_N = np.max(hist) #max of the histogram
    rrange = mmax - mmin #range of the flattened signal
    initial_guess = [mmax_N, mmin+rrange/4, rrange/2, mmax_N, mmin+3/4*rrange, rrange/2] #guess

    bounds=((0, mmin, 0, 0, mmin, 0), (2*mmax_N, mmax, rrange, 2*mmax_N, mmax, rrange))  #parameter bounds

    # Initial guess for the parameters
    '''
    if (number_of_measurements == 40)|(number_of_measurements == 70)|(number_of_measurements == 90):
        initial_guess = [0.5, 0, flattened_signal.std(), 0.5,0.001, flattened_signal.std()]
    elif number_of_measurements == 100:
        initial_guess = [100, 0, flattened_signal.std(), 1000,0.001, flattened_signal.std()]
    else:
        initial_guess = [0.5, flattened_signal.min(), flattened_signal.std(), 0.5, flattened_signal.max(), flattened_signal.std()]
    '''
    # Fit the bimozdal Gaussian curve to the histogram
    params, covariance = curve_fit(bimodal_gaussian, bin_edges[:-1], hist, p0=initial_guess, bounds=bounds) 

    # Extract the parameters for singlet and triplet peaks
    A1, mu1, sigma1, A2, mu2, sigma2 = params
    # Calculate the threshold as the average of mu1 and mu2
    #print(mu1, mu2)
    threshold = (mu1 + mu2) / 2

    #control treshhold line
    #plt.vlines(threshold,0,1000)
    # Create a thresholded matrix (0 or 1) based on the threshold
    return np.where(raw_signal_matrix > threshold, 1, 0)


def get_tau_matrix(ID_dBz, path):
    '''
    Get the tau matrix from the file
    '''

    ID_FID_time = ID_dBz-1
    return np.load(path+f"FID_time_{ID_FID_time}.npy")
   

def get_freq_array(ID_dBz, path):
    '''
    Get freq array from the file
    '''
    return np.load(path+f"delta_Bz_{ID_dBz}.npy")

def check_data(id, path, c_max = None, left=0, right="all", shots_used = "all", 
               operation_time = 0, alpha = 0.28, beta = 0.45, invert_single_shots=False,
               adaptive_prior = True):
    '''
    Process the data from the given experiment with id
    ----------------
    Parameters:
    id: int
        ID of the experiment
    filter: bool
        Whether to filter the data based on the singlet fraction
    left: int
        Left index of the data to be used
    right: int
        Right index of the data to be used
    shots_used: int
        Number of shots to be used
    top: int
        Operation time, i.e. time between realisations in ns
    Returns:
    tau_array: np.array[shot, realisation]
        Matrix of evolution times
    sigma_array: np.array[shot, realisation]
        Matrix of standard deviations
    pdfs: np.array[realisation, shot, freq]
        Matrix of pdfs
    OPX_pdf: np.array[realisation, freq]
        Matrix of pdfs from OPX
    single_shots_array: np.array[shot, realisation]
        Matrix of single shots
    mu_array: np.array[shot, realisation]
        Matrix of estimated fields
    filter: bool
        Whether the data was filtered
    '''
    #Imports
    tau_array= get_tau_matrix(id, path)[:shots_used,left:right] #import taus
    single_shots_array = get_binary_signal(id,path, number_of_measurements=shots_used)[:shots_used,left:right] #import single shots
    freq_array = get_freq_array(id,path) #import freqs
    
    if invert_single_shots:
        print("inverting")
        single_shots_array = (single_shots_array  +1) %2
    # Define the estimation method and noise model    
    N_shots, N_realisations = single_shots_array.shape
    mock_method = em.Adaptive_time_T2(freqs_grid=freq_array, 
                                    alpha_beta = (alpha, beta),
                                    N_shots = N_shots,
                                    cutoff_time = None,
                                    coeff = None,
                                    T2 = 1e99,
                                    adaptive_prior = adaptive_prior)
    noise = nm.OrnsteinUhlenbeck(sigma = 40, tc = 1/1.1*1e9)
    
    # Prepare arrays
    sigma_array = np.zeros((N_shots+1,N_realisations))
    mu_array = np.zeros((N_shots+1,N_realisations))
    cs_array = np.zeros((N_shots+1, N_realisations), dtype=float)
    pdfs = np.zeros((N_realisations, N_shots+1, len(mock_method.pdf)))    

    # Run the estimation method
    #mock_method.init_pdf()
    mock_method.init_pdf()
    for r in range(N_realisations):
        if r > 0:
            mock_method.reset_pdf(operation_time, noise)  #Diffuse the pdf between the realisations
        for s in range(N_shots):
            bit = single_shots_array[s,r] 
            tau = tau_array[s,r]
            pdfs[r,s] = mock_method.pdf
            sigma_array[s,r] = mock_method.get_std()
            mu_array[s,r] = mock_method.get_estimate(mock_method.pdf)
            #updtate
            mock_method.update_pdf(bit, tau)
        pdfs[r,s+1] = mock_method.pdf
        sigma_array[s+1,r] = mock_method.get_std()
        mu_array[s+1,r] = mock_method.get_estimate(mock_method.pdf)        
    
    #Fill cs
    cs_array[1:,:] = 1/tau_array[:,:]/sigma_array[:-1,:]*1e3  #1e3
    
    #filter all realisations of cs where cs goes above c_max
    filter_map = cs_array.max(axis=0) < c_max

    # Compare against pdf generated by OPX
    OPX_pdf = np.load(path+f"probability_Bz_{id}.npy")
    OPX_pdf = OPX_pdf[:,left:right][:,filter_map]

    #filter
    tau_array = tau_array[:,filter_map]
    sigma_array = sigma_array[:,filter_map]
    pdfs = pdfs[filter_map,:,:]
    single_shots_array = single_shots_array[:,filter_map]
    mu_array = mu_array[:,filter_map]
    cs_array = cs_array[:,filter_map]

    return tau_array, sigma_array, pdfs, OPX_pdf, single_shots_array, mu_array, cs_array
    


def plot_data(id, tau_array, sigma_array, pdfs, OPX_pdf, top, c_max, add2name=""):
    '''
    Creates the plot 3x2 of the data
    First row: c coefficient vs shots vs realisations, histogram of c coefficient
    Second row: final pdf from reverse engineering, final pdf from OPX
    Third row: sigma vs shots vs realisations, average sigma vs shots

    Args:
    id: int
        ID of the experiment
    tau_array: np.array[shot, realisation]
        Matrix of evolution times
    sigma_array: np.array[shot, realisation]
        Matrix of standard deviations
    pdfs: np.array[realisation, shot, freq]
        Matrix of pdfs
    OPX_pdf: np.array[realisation, freq]
        Matrix of pdfs from OPX
    top: float
        Operation time, i.e. time between realisations in ns
    c_max: float
        Maximum value of c coefficient
    add2name: str
        Additional string to be added to the name of the file
    '''

    
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.5, left=0.08,right=0.98,top=0.93, bottom=0.1)
    final_pdf = pdfs[:,-1,:]
    
    N_shots = tau_array.shape[0]
    N_realisations = tau_array.shape[1]

    #one title for all plots
    if filter==True:
        plt.suptitle(r"N="+str(N_shots)+" , ID = "+str(id)+" $T_{op}$"+str(top/1e6)+"ms, FILTERED")
    else:
        plt.suptitle(r"N="+str(N_shots)+" , ID = "+str(id)+" $T_{op}$"+str(top/1e6)+"ms, RAW")

    # Plot (0,0) the c coefficient vs shots vs realisations
    axs[0,0].set_title("Was c coefficient constant in ID"+str(id))
    cs = 1/tau_array[1:,:]/sigma_array[:-1,:]*1e3
    
    im = axs[0,0].contourf(1/tau_array[1:,:]/sigma_array[:-1,:]*1e3)
    cb = plt.colorbar(im, ax = axs[0,0])
    cb.set_label("c")

    # Plot (0,1) histogram of c coefficient 
    axs[0,1].hist(cs.flatten(), bins = 100)
    axs[0,1].set_xlabel("c")
    

    # Plot(1,0) final pdf from reverse engineering
    final_pdf = final_pdf.T/final_pdf.T.max(axis=0)
    axs[1,0].imshow(final_pdf, interpolation='none', aspect='auto', origin='lower')
    axs[1,0].set_title("Reverse engineered final PDF,N="+str(N_shots)+" ID: "+str(id))
    #plt.xlim(1000,4000)
    axs[1,0].set_xlabel("Realisations")
    axs[1,0].set_ylabel(r"$\Delta B_z$ (MHz)")
    
    # Plot(1,1) final pdf from OPX
    OPX_pdf = OPX_pdf/OPX_pdf.max(axis=0)
    axs[1,1].set_title("OPX final PDF, N="+str(N_shots)+" ID: "+str(id))
    axs[1,1].imshow(OPX_pdf,interpolation='none', aspect='auto', origin='lower')
    axs[1,1].set_xlabel("Realisations")
    axs[1,1].set_ylabel(r"$\Delta B_z$ (MHz)")

    # Plot(2,0) sigma vs shots vs realisations
    X,Y = np.meshgrid(np.arange(N_realisations),np.arange(N_shots))
    im= axs[2,0].contourf(X,Y,np.log10(sigma_array), levels=100)
    axs[2,0].set_ylabel("shots")
    axs[2,0].set_xlabel("realisations")
    cb = plt.colorbar(im, ax = axs[2,0])
    cb.set_label("log10(sigma)")
    axs[2,0].set_title("Reconstructed $\sigma_n$, N="+str(N_shots))
    
    # Plot(2,1) average sigma vs shots
    axs[2,1].errorbar(np.arange(N_shots), np.average(sigma_array,axis=1), yerr=np.std(sigma_array,axis=1),fmt="o", label="average")
    axs[2,1].set_ylabel("$\sigma$ (MHz)")
    axs[2,1].set_xlabel("shots")
    axs[2,1].grid(True)

    #Save figure
    plt.savefig("figures/adaptive_top_"+str(top/1e6)+"ms_N_"+str(N_shots)+"_"+str(id)+"_cmax_"+str(c_max)+"_"+add2name+".png", dpi=100)

    plt.close()

def plot_diagnosis(id,tau_array, sigma_array, pdfs, OPX_pdf, single_shots_array, 
                   mu_array, c_max, top=0, add2name=""):
    '''
    Creates the plot 3x2 of the data
    First row: times used in the experiment, computed std sigmas
    Second row: array of single shots, estimated field shot by shot
    Third row: reconstructed PDF, OPX PDF

    Args:
    id: int
        ID of the experiment
    tau_array: np.array[shot, realisation]
        Matrix of evolution times
    sigma_array: np.array[shot, realisation]    
        Matrix of standard deviations
    pdfs: np.array[realisation, shot, freq]
        Matrix of pdfs
    OPX_pdf: np.array[realisation, freq]
        Matrix of pdfs from OPX
    single_shots_array: np.array[shot, realisation]
        Matrix of single shots
    mu_array: np.array[shot, realisation]
        Matrix of estimated fields
    c_max: float
        Maximum value of c coefficient
    top: float
        Operation time, i.e. time between realisations in ns
    add2name: str
        Additional string to be added to the name of the file
    

    '''

    fig, ax = plt.subplots(3, 2, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.5, left=0.08,right=0.98,top=0.93, bottom=0.1)
    N_shots = single_shots_array.shape[0]
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    final_pdf = pdfs[:,-1,:]
    
    if filter==True:
        plt.suptitle(r"N="+str(N_shots)+" , ID = "+str(id)+" $T_{op}$"+str(top/1e6)+"ms, Filtered")
    else:
        plt.suptitle(r"N="+str(N_shots)+" , ID = "+str(id)+" $T_{op}$"+str(top/1e6)+"ms, RAW")

    #[0,0] times used in the experiment
    im = ax[0,0].pcolormesh(tau_array)
    cb = plt.colorbar(im, ax=ax[0,0])
    cb.set_label(r"$\tau$ (ns)")
    
    #[0,1] computed std sigmas
    im = ax[0,1].pcolormesh(sigma_array)
    cb = plt.colorbar(im, ax=ax[0,1])
    cb.set_label(r"$\sigma$ (MHz)")
    #flip the y axis of the imshow of single shots array and make it square
    
    #[1,0] array of single shots
    ax[1,0].set_title("Single shot data")
    im = ax[1,0].imshow(single_shots_array, interpolation='none', aspect='auto', cmap='binary_r', origin='lower')

    #[1,1] estimated field shot by shot
    im = ax[1,1].pcolormesh(mu_array)
    cb = plt.colorbar(im, ax=ax[1,1])
    cb.set_label("$\mu$ (MHz)")

    #[2,0] reconstructed PDF
    normed = final_pdf.T/final_pdf.T.max(axis=0)
    ax[2,0].set_title("Reconstructed PDF, N="+str(N_shots)+" ID: "+str(id))
    ax[2,0].contourf(normed, levels=100)
    #devide each column of final_pdf by the max of the column
    ax[2,0].set_xlabel("Realisations")
    ax[2,0].set_ylabel(r"$\Delta B_z$ (MHz)")

    #[2,1] OPX PDF
    ax[2,1].set_title("OPX final PDF, N="+str(N_shots)+" ID: "+str(id))
    #devide each column of prob_matrix by the max of the column
    norm2 = OPX_pdf/OPX_pdf.max(axis=0)
    ax[2,1].contourf(norm2, levels=100)
    ax[2,1].set_xlabel("Realisations")
    ax[2,1].set_ylabel(r"$\Delta B_z$ (MHz)")
    plt.savefig("figures/adaptive_top_"+str(top/1e6)+"ms_N_"+str(N_shots)+"_"+str(id)+"_cmax_"+str(c_max)+"_"+add2name+"_diag.png", dpi=100)
    plt.close()


def plot_pdf_evolution(id, pdfs, realisations, top, Nshots):
    '''
    Plots the evolution of the pdfs for the given realisations
    Args:
    id: int
        ID of the experiment
    pdfs: np.array[realisation, shot, freq]
        Matrix of pdfs 
    realisations: list
        List of realisations to be plotted
    top: float
        Operation time, i.e. time between realisations in ns
    Nshots: int
        Number of shots used
    '''
    fig, axs = plt.subplots(1, len(realisations), figsize=(10, 1))
    #in each subplots put a contour plot of the pdfs[r] for 10 indeces of r
    axs[0].set_ylabel(r"$\Delta B_z$ (MHz)")

    for i,r in enumerate(realisations):
        normed = pdfs[r].T/pdfs[r].T.max(axis=0)
        im = axs[i].imshow(pdfs[r].T, interpolation='none', origin='lower', aspect='auto', cmap='jet')
        axs[i].set_title("r = "+str(r))
        axs[i].set_xlabel("shots")
        if i>0:
            axs[i].set_yticks([])

    plt.savefig("figures/raw_"+str(top)+"_N"+str(Nshots)+"_ID"+str(id)+"_realisations_"+str(
                            realisations[0])+"_"+str(realisations[-1])+".png", dpi=100)
    plt.close()
    


def get_estimation_from_shulman(methods, N_shots_used, N_realisations, single_shots_array, freq_array,
                                cycle_time, dt_realisation, noise):
    
    '''
    Get the estimation from the Shulman method
    Args:
    methods: list
        List of estimation methods
    N_shots_used: int
        Number of shots used
    N_realisations: int
        Number of realisations
    single_shots_array: np.array[shot, realisation]
        Matrix of single shots
    freq_array: np.array[freq]
        Array of frequencies
    cycle_time: float   
        Time between shots in ns    
    dt_realisation: float
        Time between realisations in ns
    noise: noise model
        Noise model
    Returns:
    sigma_array: np.array[estimation method, shot, realisation] 
        Matrix of standard deviations
    mu_array: np.array[estimation method, shot, realisation]    
        Matrix of estimated fields
    n_taus_used: np.array[estimation method, shot, realisation] 
        Matrix of number of taus used
    taus_used: np.array[estimation method, shot, realisation]   
        Matrix of taus used
    final_pdf: np.array[estimation method, realisation, freq]   
        Matrix of final pdfs
    final_mu: np.array[estimation method, realisation]
        Matrix of final estimates
    '''
    
    #Preparing arrays
    N_shots = single_shots_array.shape[0]
    taus = np.linspace(0,N_shots-1,N_shots)
    sigma_array = np.zeros((len(methods), N_shots_used, N_realisations))
    mu_array = np.zeros((len(methods), N_shots_used, N_realisations))
    n_taus_used= np.zeros((len(methods), N_shots_used, N_realisations))
    taus_used = np.zeros((len(methods), N_shots_used, N_realisations))-1
    final_pdf = np.zeros((len(methods), N_realisations, len(freq_array)))

    #Get the real field from the last method (shulman)
    method = methods[-1]
    method.init_pdf() #initialize the pdf
    final_mu = np.zeros(N_realisations)
    
    for r in range(N_realisations):
        if r>0:
            method.diffuse_pdf(dt_realisation, noise) #diffuse the pdf between the realisations
        for s in range(N_shots):
            tau = method.next_tau(s) #get the next tau
            bit = single_shots_array[s,r] #get the next bit
            method.update_pdf(bit, tau) #update the pdf
            method.diffuse_pdf(cycle_time, noise) #diffuse the pdf between the shots
            #n_taus_used[-1,s,r] = 1
        final_pdf[-1,r,:] = method.get_pdf(freq_array) #get the final pdf
        final_mu[r] = method.get_estimate(method.pdf) #get the final estimate
    

    #Estimate using the other methods
    dt_realisation = dt_realisation + (100-N_shots_used)*cycle_time 
    for mn,method in enumerate(methods[:-1]):
        method.init_pdf()
        for r in range(N_realisations):
            if r>0:
                method.reset_pdf(dt_realisation, noise)
            s0 = 0
            for s in range(N_shots_used):
                tau = method.next_tau(s)
                n = int(np.floor(tau))
                if n>99: 
                    #print('over')
                    n = 99

                # if tau was already used
                if taus[n] in taus_used[mn,:,r]:
                    sigma_array[mn,s,r] = 0
                    mu_array[mn,s,r] = final_mu[r]
                    taus_used[mn, s, r] = 0
                    n_taus_used[mn,s,r] = 0 #fill this matrix with 0 to not count into average
                
                # if this is a new tau
                else:
                    taus_used[mn, s, r] = taus[n] #save tau
                    bit = single_shots_array[n,r] #get the next bit
                    method.update_pdf(bit, tau) #update the pdf
                    sigma_array[mn,s,r] = method.get_std() #get the std
                    mu_array[mn,s,r] = method.get_estimate(method.pdf) #get the estimate
                    n_taus_used[mn,s,r] = 1 #fill this matrix with 1 to count into average

                method.diffuse_pdf(cycle_time, noise)
            final_pdf[mn,r,:] = method.get_pdf(freq_array)

    return sigma_array, mu_array, n_taus_used, taus_used, final_pdf, final_mu