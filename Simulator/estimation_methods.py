# %%
import numpy as np
from typing import Union, List
import constants_and_units as cu
import scipy.stats as stats
import random

# %%
class Estimation_method():
    '''
    This is a dummy class. It specifies the interface for estimation methods. One has to specify:
    Functions
    -------------
    init_pdf -- returns the initial pdf
    reset_pdf -- resets the pdf after the previous and before the next realisation (apriori)
    next_tau -- returns the next evolution time tau
    update_pdf -- updates the pdf after the shot
    if_terminate -- checks if the estimation protocol should be terminated
    get_estimate -- returns the estimate of the frequency
    get_avg_value -- returns the avg of the pdf
    get_std_value -- returns the standard deviation of the pdf
    '''

    def __str__(self):
        return "Estimation_method"

    def __init__(self, params):
        '''
        Constructor that loads the (hyper)parameters of the estimation method
        Arguments
        ------------
        params : dict
            dictionary of parameters
        '''
        self.params = params
	

    def init_pdf(self):
        '''
        This function returns the initial pdf
        Default: flat pdf
        '''
        pass
        

    def reset_pdf(self, dt_realisations, noise):
        '''
        This function creates the a priori pdf using posteriori from the previous
        Default: init_pdf()
        '''
        return self.init_pdf()

    def next_tau(self,  n):
        '''
        This function returns the next tau using that depends on the current pdf/its moments or iteration number n
        '''
        tau = 1
        return tau
    
    def update_pdf(self, bit, tau, confidence):
        '''
        This function updates the probability distribution after the shot
        Default -> does nothing with the pdf
        '''
        pass

    def if_terminate(self, n,  time):
        '''
        This function checks if the estimation protocol should be terminated
        Default -> does nothing
        '''
        return False

   
  

    def get_avg(self, pdf):
        if pdf == None:
            pass #f(self.pdf)
        else:
            pass #f(pdf)

    def get_std(self, pdf):
        if pdf == None:
            pass #f(self.pdf)
        else:
            pass #f(pdf)

    def get_pdf(self):
        return self.pdf

    def get_estimate(self, pdf):
        if pdf is None:
            self.get_avg()
        else:
            self.get_avg(pdf)
        return
    
    def get_pdf(self, grid):
        return self.pdf


# %%
class Linear_time(Estimation_method):
    '''
    Class that definees the shulman estimation method
    Features:
    -----------
    Prior distribution: flat/adaptive
            (uniform distribution on the freq grid)/(fokker-planck equation)
    Evolution time: linear 
            (passed by the taus args)

    Attributes
    ------------
    freqs_grid : array
        grid of frequencies
    taus : array 
        list of  evolution times taus
    alpha : float
         alpha parameter
    beta : float
         beta parameter
    Functions
    -----------
    See Estimation_method
    '''
    

    def __init__(self, freqs_grid: List[float], taus: List[float], alpha_beta: List[float], 
                 adaptive_prior = False, T2 = 1e999, add_str = ""):
        super().__init__(None)
        self.freqs = freqs_grid
        self.alpha = alpha_beta[0]
        self.beta = alpha_beta[1]
        self.T2 =  T2
        self.taus = taus
        self.adaptive_prior = adaptive_prior
        self.init_pdf()
        self.add_str = add_str
        

    def __str__(self):
        if self.adaptive_prior:
            return "Linear_time_adaptive_prior"+self.add_str
        else:
            return "Linear_time"+self.add_str
    
    def init_pdf(self):
        '''
        This function initializes the pdf
        In this case it is just a flat pdf that uses the grid of frequencies
        '''
        pdf = np.ones(len(self.freqs))
        pdf /= np.sum(pdf)
        self.pdf = pdf


    def reset_pdf(self, dt_realisations, noise):
        '''
        This function resets the pdf after the realisation
        In this case it is just a flat pdf
        '''
        #print(dt_realisations)
        if self.adaptive_prior:
            #pre = self.get_std(self.pdf)
            #print("pre"+str(self.get_std(self.pdf)))
            self.diffuse_pdf(dt_realisations, noise)
            #print("post"+str(self.get_std(self.pdf)))
            #print("post"+str(np.sqrt(50**2 + (pre**2 - 50s**2)*np.exp(-2*dt_realisations/noise.get_tc()))))
        else:

            self.init_pdf()
        

    def diffuse_pdf(self, dt_realisations, noise):
        '''
        This function resets the pdf after the realisation
        In this case we follow fokker-planck equation and update the avg and std of the pdf
        Parameters
        ------------
        dt_realisations : float
            time between realisations
        noise : instance of Noise class
            noise process   
        
        Returns
        ------------
        pdf : array
            apriori probability distribution
        '''
        avg, std = self.get_avg(), self.get_std()
        sig, tc = noise.get_sigma(), noise.get_tc()
        avg *= np.exp(-dt_realisations/tc)
        avg += noise.om0
        #print(noise.om0)
        std = np.sqrt(sig**2+(std**2-sig**2)*np.exp(-2*dt_realisations/tc))
        pdf = np.exp(-(self.freqs-avg)**2/(2*std**2))
        self.pdf =  pdf/np.sum(pdf)

    def next_tau(self,n):
        '''
        This function returns the next tau using predefined list of taus and iteration number n
        '''
        return self.taus[n]
    
    def update_pdf(self,bit, tau, confidence = 1):
        '''
        This function updates the probability distribution after the shot
        Parameters
        ------------
        bit : bool
         result of the shot (0 or 1)
        tau : float
         evolution time
        pdf : array
            current probability distribution
        confidence : float
            confidence of the measurement (default = 1)
        Returns
        ------------
        posterior_pdf : array
            updated probability distribution
        '''
        q = 2*bit-1
        p = 1/2+q*1/2*(self.alpha+self.beta*np.cos(self.freqs*tau*cu.f2omGHz)*np.exp(-tau/self.T2))	
        posterior_pdf = self.pdf*p
        posterior_pdf /= np.sum(posterior_pdf)
        self.pdf = posterior_pdf
    
    def if_terminate(self, n, time):
        '''
        This function checks if the estimation protocol should be terminated
        Parameters
        ------------
        n : int
            iteration number
        Returns
        ------------
        bool
            True if the protocol should be terminated
        '''
        return n<len(self.taus)
    
    def get_avg(self, pdf=None):
        '''
        This function returns the average frequency using the current pdf and the grid of frequencies
        '''
        
        if pdf is not None:    
            return np.sum(pdf*self.freqs)
        else:
            return np.sum(self.pdf*self.freqs)
    

    def get_std(self, pdf = None):
        '''
        This function returns the standard deviation of frequency distribution 
        using the current pdf and the grid of frequencies
        '''
        if pdf is not None:
            return np.sqrt(np.sum(pdf*self.freqs**2)-np.sum(pdf*self.freqs)**2)
        else:
            return np.sqrt(np.sum(self.pdf*self.freqs**2)-np.sum(self.pdf*self.freqs)**2)
        

    def get_estimate(self, pdf):
        return self.get_avg(pdf)       

# %%
class Adaptive_time(Linear_time):
    '''
    Class that definees the estimation method with adaptive time with flat prior
    Features:
    -----------
    Prior distribution: flat/adaptive
            (uniform distribution over the grid of frequencies)/(diffused pdf from the previous iteration)
    Evolution time: adaptive 
            (the next time taken according to the inverse of the standard deviation of the pdf)
    Arguments:
    ------------
    freqs_grid : array
        grid of frequencies
    alpha : float
        alpha parameter
    beta : float
        beta parameter
    Functions
    -----------
    See Estimation_method
    '''


    def __init__(self, freqs_grid, alpha_beta, N_shots, cutoff_time, coeff, 
                 adaptive_prior=False, T2 = 1e9, add_str = ""):
        super().__init__(freqs_grid=freqs_grid, taus = None, alpha_beta=alpha_beta, 
                         adaptive_prior=adaptive_prior, T2=T2, add_str=add_str)
        self.N = N_shots
        self.cutoff_time = cutoff_time
        self.coeff = coeff

    def __str__(self):
        if self.adaptive_prior:
            return "Adaptive_time_adaptive_prior_c_"+str(self.coeff)+self.add_str
        else:
            return "Adaptive_time_c_"+str(self.coeff)+self.add_str


    def if_terminate(self, n, time):
        return n<self.N


    def next_tau(self,n):
        '''
        This function returns the next tau using predefined list of taus and iteration number n
        '''
        avg, std = self.get_avg(), self.get_std()
        if np.isnan(std):
            print(self.pdf)
        adaptive_tau = int(1/(self.coeff*std*1e-3))
        if adaptive_tau < self.cutoff_time:
            return adaptive_tau
        else:
           return self.cutoff_time


class Adaptive_time_OPX(Adaptive_time):


    def get_std(self, pdf=None, assert_overflow=False):
        
        mean_var = self.get_avg(pdf=pdf)
        comp_var = 0 # "C" in Kahan wikipedia implementation
        variance = 0 # "sum" in Kahan wikipedia implementation
        
        if pdf is not None:
            for i in range(len(pdf)):
                delta_var = self.freqs[i]-mean_var
                term_qua = pdf[i]*(delta_var**2-comp_var) # "y" in Kahan wikipedia implementation
                temp_qua = variance + term_qua # "t" in Kahan wikipedia implementation
                comp_var = (temp_qua-variance) - term_qua
                variance = temp_qua
                
        else:
            for i in range(len(self.pdf)):
                delta_var = self.freqs[i]-mean_var
                term_qua = self.pdf[i]*(delta_var**2-comp_var) # "y" in Kahan wikipedia implementation
                temp_qua = variance + term_qua # "t" in Kahan wikipedia implementation
                comp_var = (temp_qua-variance) - term_qua
                variance = temp_qua
        
        
        
        # Check if 8/sqrt(1000*variance_qua) overflows (larger than 8)
        if assert_overflow:
            # OPX works with a scaling factor of 10000 on the variance in order to avoid overflow
            variance_qua = variance / 10000
            cc = 8 # c = 100/cc, so cc=8 -> c=12.5
            assert cc / np.sqrt(1000*variance_qua) < 8, "OPX Overflow!"

        return np.sqrt(variance)


# %%
class Random_time_window(Adaptive_time):
    '''
    Heuristic random estimation method
    Features:
    -----------
    Prior distribution: flat/adaptive
            (uniform distribution over the grid of frequencies)/(adaptive)
    Evolution time: random
            (drawn from the same distribution)

    Arguments
    ------------
    window : tuple
        (min, max) of the time window
    freqs_grid : array
        grid of frequencies
    alpha : float
        alpha parameter
    beta : float
    '''

    
    def __init__(self, times_window, freqs_grid, alpha_beta,
                  N_shots, adaptive_prior=False ,add_str = ""):
        super().__init__(freqs_grid, alpha_beta = alpha_beta, 
                         N_shots = N_shots, adaptive_prior=adaptive_prior,
                         cutoff_time=None, coeff=None, add_str=add_str)
        self.time_min = times_window[0]
        self.time_max = times_window[1]
        self.str = "Random_time_window"

        self.drawn_times = random.sample(range(self.time_min, self.time_max+1), N_shots)
        
    def __str__(self):
        if self.adaptive_prior:
            return "Random_time_adaptive_prior"
        else:
            return "Random_time"

    def reset_pdf(self, dt_realisations, noise):
        '''
        This function resets the pdf after the realisation
        In this case it is just a flat pdf
        '''
        #print(dt_realisations)
        self.drawn_times = random.sample(range(self.time_min, self.time_max+1), self.N)
        if self.adaptive_prior:
            #pre = self.get_std(self.pdf)
            #print("pre"+str(self.get_std(self.pdf)))
            self.diffuse_pdf(dt_realisations, noise)
            #print("post"+str(self.get_std(self.pdf)))
            #print("post"+str(np.sqrt(50**2 + (pre**2 - 50s**2)*np.exp(-2*dt_realisations/noise.get_tc()))))
        else:
            
            self.init_pdf()


    def next_tau(self, n):
        return self.drawn_times[n]


# %%
class Random_time_two_windows(Random_time_window):
    '''
    Heuristic random estimation method
    Features:
    -----------
    Prior distribution: flat/adaptive
            (uniform distribution over the grid of frequencies)/(adaptive)
    Evolution time: random
            (drawn from the same distribution)

    Arguments
    ------------
    window : tuple
        (min, max) of the time window
    freqs_grid : array
        grid of frequencies
    alpha : float
        alpha parameter
    beta : float
    '''

    
    def __init__(self, even_times_window, odd_times_window, 
                freqs_grid, alpha_beta, N_shots, adaptive_prior=False):
        super().__init__([0,0],freqs_grid, alpha_beta = alpha_beta, N_shots = N_shots, adaptive_prior=adaptive_prior)
        self.even_time_min = even_times_window[0]
        self.even_time_max = even_times_window[1]
        self.odd_time_min = odd_times_window[0]
        self.odd_time_max = odd_times_window[1]


    
    def __str__(self):
        if self.adaptive_prior:
            return "Random_time_two_windows_adaptive_prior"+self.add_str
        else:
            return "Random_time_two_windows"+self.add_str

    def next_tau(self, n):
        if n %2 ==0:
            return np.random.uniform(self.even_time_min, self.even_time_max)
        else:
            return np.random.uniform(self.odd_time_min, self.odd_time_max)

# %%
        

# %%


# %%
class Chi_square(Estimation_method):
    '''
    Class that definees the estimation method with adaptive time with fixed prior
    Features:
    -----------
    Prior distribution: fixed mu=mu0, sigma=sig0 
            (the same for each realisaiton)
    Evolution time: adaptive chi2
            (time that maximizes the decrease in sigma after next single shot)
    
    Arguments
        ------------
        sig0 : float
            initial standard deviation of the frequency distribution
        mu0 : float
            initial expected frequency (default = 0)
        dephasing_time : float
            maximum evolution time of the qubit (typically T = 1e5 ns)
        N_shots : int
            number of shots
    Functions
    -----------
    See Estimation_method
    '''
 
    def __init__(self, alpha_beta, sig0, mu0, dephasing_time, N_shots, freqs_grid,
                 adaptive_prior = False, add_str=""):
        '''
        Constructor that loads the (hyper)parameters of the estimation method
        '''
        self.sig0 = sig0
        self.mu0 = mu0
        self.T = dephasing_time
        self.N = N_shots
        self.adaptive_prior = adaptive_prior
        self.alpha = alpha_beta[0]
        self.beta = alpha_beta[1]
        self.add_str = add_str
        self.freqs = freqs_grid 
        
        
    def __str__(self):
        if self.adaptive_prior:
            return "Chi_square_adaptive_prior"+self.add_str
        else:
            return "Chi_square"+self.add_str


    def init_pdf(self):
        '''
        This function returns the initial pdf
        Default: flat pdf
        '''
        self.pdf = {"mu":self.mu0, "sig":self.sig0}


    def reset_pdf(self, dt_realisations, noise):
        if self.adaptive_prior:
            self.diffuse_pdf(dt_realisations,noise)
        else:
            self.init_pdf()

    def diffuse_pdf(self, dt_realisations, noise):
        avg, std = self.get_avg(), self.get_std()
        sig, tc = noise.get_sigma(), noise.get_tc()
        avg *= np.exp(-dt_realisations/tc)
        std = np.sqrt(sig**2+(std**2-sig**2)*np.exp(-2*dt_realisations/tc))
        self.pdf = {"mu": avg, "sig": std}


    def next_tau(self, n):
        '''
        This function returns the next tau using that depends on the current pdf/its moments or iteration number n
        '''
        mu_omGHz = self.pdf["mu"]*cu.f2omGHz
        sig_omGHz = self.pdf["sig"]*cu.f2omGHz
        tau = (2/self.T**2+sig_omGHz**2)**(-1/2)
        
        if 2*np.pi*tau*mu_omGHz > np.pi/2:
            k = max(0,np.floor(tau*mu_omGHz/np.pi-1/2))
            tau = (k+1/2)*np.pi/mu_omGHz
    
        return tau 
    
    
    
    def update_pdf(self, bit, tau, confidence=1):
        '''
        This function updates the probability distribution after the shot
        Default -> does nothing with the pdf
        '''
        mu_omGHz = self.pdf["mu"]*cu.f2omGHz
        sig_omGHz = self.pdf["sig"]*cu.f2omGHz

        bit = 1-bit
        E = np.exp(-tau**2*(sig_omGHz**2/2+1/self.T**2))
        q = (2*bit - 1)
        C = np.cos(mu_omGHz*tau)
        P = 1/2-q*1/2*(self.alpha+E*C)	
        #P = 1/2*(1 - q*E*C)
    
        y = (1/2/P)*(
            mu_omGHz**2+sig_omGHz**2+q*E*(
                                2*mu_omGHz*sig_omGHz**2*tau*np.sin(mu_omGHz*tau
                                )-(mu_omGHz**2+sig_omGHz**2 - sig_omGHz**4*tau**2
                                )*np.cos(mu_omGHz*tau)))
        y2 = (1/P/2)*(
            mu_omGHz**4+ 6*mu_omGHz**2*sig_omGHz**2+3*sig_omGHz**4+q*E*(
            4*mu_omGHz*sig_omGHz**2*tau*(
            mu_omGHz**2+3*sig_omGHz**2-sig_omGHz**4*tau**2
            )*np.sin(mu_omGHz*tau) - (
            mu_omGHz**4+mu_omGHz**2*(
            6*sig_omGHz**2-6*sig_omGHz**4*tau**2
            )+ sig_omGHz**4*(
            3-6*sig_omGHz**2*tau**2+sig_omGHz**4*tau**4
            ))*np.cos(mu_omGHz*tau)
            ))
            
        s = np.sqrt(max(0, 3*y**2/2.-y2/2.))    
        mu = np.sqrt(s)/cu.f2omGHz
        sig = np.sqrt(y - s)/cu.f2omGHz
        self.pdf = {"mu":mu, "sig":sig}


    def if_terminate(self, n, time):
        '''
        This function checks if the estimation protocol should be terminated
        Default -> does nothing
        '''
        return n<self.N
  

    def get_avg(self, pdf = None):
        if pdf is None:
            return self.pdf["mu"]
        else:
            return self.pdf["mu"]

    def get_std(self, pdf = None):
        if pdf is None:
            return self.pdf["sig"]
        else:
            return self.pdf["sig"]
    

    def get_estimate(self, pdf):
        return self.get_avg(pdf)
    

    def get_pdf(self, freq_grid):
        mu = self.pdf["mu"]
        sig = self.pdf["sig"]
        pdf =  np.exp(-(freq_grid-mu)**2/(2*sig**2))
        return pdf/np.sum(pdf)

    

# %%
class Inverse_mu(Adaptive_time):
    '''
    Heuristic random estimation method
    Features:
    -----------
    Prior distribution: flat/adaptive
            (uniform distribution over the grid of frequencies)/(adaptive)
    Evolution time: random
            (drawn from the same distribution)

    Arguments
    ------------
    window : tuple
        (min, max) of the time window
    freqs_grid : array
        grid of frequencies
    alpha : float
        alpha parameter
    beta : float
    '''

    
    def __init__(self, fuzz, t0, N_shots, freqs_grid, alpha_beta, adaptive_prior=False):
        super().__init__(freqs_grid, alpha_beta = alpha_beta, N_shots = N_shots, adaptive_prior=adaptive_prior)
        self.t0 = t0
        self.N = N_shots
        self.fuzz = fuzz

    
    def __str__(self):
        if self.adaptive_prior:
            return "Inverse_mu_adaptive_prior"+self.add_str
        else:
            return "Inverse_mu"+self.add_str

    def next_tau(self, n):
        if n<2:
            return self.t0*n
        else:
            return 1e3/self.get_estimate(self.pdf) + self.fuzz*np.random.randn()


class Adaptive_time_T2(Adaptive_time):
      
            
    def __init__(self, T2, N_shots, freqs_grid, alpha_beta, coeff, cutoff_time, adaptive_prior=False):
        super().__init__(freqs_grid, alpha_beta = alpha_beta, N_shots = N_shots, coeff=coeff, 
                          cutoff_time = cutoff_time,
                          adaptive_prior=adaptive_prior)
        self.T2 = T2




    def update_pdf(self,bit, tau, confidence = 1):
        '''
        This function updates the probability distribution after the shot
        Parameters
        ------------
        bit : bool
         result of the shot (0 or 1)
        tau : float
         evolution time
        pdf : array
            current probability distribution
        confidence : float
            confidence of the measurement (default = 1)
        Returns
        ------------
        posterior_pdf : array
            updated probability distribution
        '''
        q = 2*bit-1
        p = 1/2+q*1/2*(self.alpha+self.beta*np.cos(self.freqs*tau*cu.f2omGHz)*np.exp(-tau/self.T2))	
        posterior_pdf = self.pdf*p
        posterior_pdf /= np.sum(posterior_pdf)
        self.pdf = posterior_pdf

    def init_pdf2(self):
        '''
        This function initializes the pdf
        In this case it is just a flat pdf that uses the grid of frequencies
        '''
        sig = 5
        mu = 25
        pdf = np.exp(-(self.freqs-mu)**2/(2*sig**2))
        pdf /= np.sum(pdf)
        self.pdf = pdf

# %%
