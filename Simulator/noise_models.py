import numpy as np
import scipy as sp
import scipy.interpolate
from typing import Union, List


class NoiseProcess:
    """
    This is a dummy function. It is used to define the interface of the noise processes
    One has to specify the update equation and how to define initial value
    """
    x: float

    def __init__(self, x0=None):
        self.x = self.set_initial_value(x0)

    def update(self, dt):
        pass

    def get_value(self):
        return self.x

    def set_initial_value(self, x0):
        self.x = x0

    def reset(self):
        pass


class OrnsteinUhlenbeck(NoiseProcess):
    """
    Class that defines the Ornstein-Uhlenbeck noise (diffusion process)
    """
    sigma: float
    tc: float

    def __init__(self, sigma, tc, x0=None):
        """
        Constructor of the class
        Parameters
        -------------
        sigma : float
          standard deviation of the noise
        tc : float
            correlation time of the noise
        om0:
            initial value of the noise (If none it is generated randomly)
        """
        self.sigma = sigma
        self.tc = tc
        self.set_initial_value(x0)

    def set_sigma_tc(self, sigma, tc):
        """
        This function sets the standard deviation and correlation time of the noise
        """
        self.sigma = sigma
        self.tc = tc

    def update(self, dt):
        """
        This function updates the noise stepsss
        dt -- time step
        TODO -- enable using fixed trajectories (seed)
        """
        self.x = (
            self.x * np.exp(-dt / self.tc)
            + np.sqrt(1 - np.exp(-2 * dt / self.tc)) * self.sigma * np.random.normal()
        )
        return self.x

    def set_initial_value(self, x0=None):
        """
        This function sets the initial value of the noise
        If x0 is None it is generated randomly
        """
        if x0 is None:
            self.x = self.sigma * np.random.normal()
        else:
            self.x = x0

    def get_sigma(self):
        """
        This function returns the standard deviation of the noise
        """
        return self.sigma

    def get_tc(self):
        """
        This function returns the correlation time of the noise
        """
        return self.tc


class SingleTrajectory:
    """
    In this function we pass or generate (generate_trajectory) the single trajectory,
    that might be common for multiple methods/trials
    Arguments
    -------------
    times : array
        array of times
    noise_process : instance of noise_process class (optional, default None)
        noise process
    interp_trajectory : function
        evaluates trajectoy at given points in times[0]<time<times[-1]
        do not mistake with the argument of constructor (array).
        This is a function that is generated by generate_trajectory or by
        interpolating the trajectory passed to the constructor
    clock : float
        current time of the trajectory

    Functions
    -------------
    generate_trajectory : generates the trajectory using the noise process
    get_value : returns the value of the trajectory at the current time
    get_sig : returns the standard deviation of the noise process
    get_tc : returns the correlation time of the noise process
    reset : resets the clock to 0
    """

    def __init__(self, times: List[float], noise_process: NoiseProcess=None, trajectory: List[float]=None):
        """
        Constructor of single trajectory class. We can either pass the
        trajectory or the field or generate it using the noise process
        Parameters
        -------------
        times : array
            array of times

        noise_process : instance of noise_process class
            noise process
        or
        trajectory : array
            trajectory at given points in time (times)
        """

        if trajectory is not None:
            self.interp_trajectory = sp.interpolate.interp1d(times, trajectory)
            self.noise_process = noise_process
        elif noise_process is not None:
            self.noise_process = noise_process
            self.generate_trajectory(times, noise_process)
        else:
            print("Error: No noise process or trajectory passed")
        self.clock = 0

    def generate_trajectory(self, times, noise_process: NoiseProcess):
        """
        This function generates the trajectory using the noise process
        it creates a FUNCTION self.trajectory with accepts any times[0]<time<times[-1] as argument
        Parameters
        -------------
        times : array
            array of times
        noise_process : instance of noise_process class
            noise process
        """

        trajectory = np.zeros(len(times))
        trajectory[0] = noise_process.get_value()
        for n, t in enumerate(times):
            if n > 0:
                trajectory[n] += noise_process.update(times[n] - times[n - 1])
        self.interp_trajectory = sp.interpolate.interp1d(times, trajectory)

    def update(self, dt):
        """
        This function updates the trajectory by dt
        Parameters
        -------------
        dt : float
            time step
        Returns
        -------------
        trajectory at time self.time+dt
        """
        self.clock += dt
        return self.interp_trajectory(self.clock)

    def get_value(self, t=None):
        """
        This function returns the value of the trajectory at time self.time
        """
        if t is None:
            return self.interp_trajectory(self.clock)
        else:
            return self.interp_trajectory(t)

    def get_sigma(self):
        return self.noise_process.sigma

    def get_tc(self):
        return self.noise_process.tc

    def reset(self):
        self.clock = 0


class Ornstein_plus_fast(OrnsteinUhlenbeck):
    def __init__(self, sigma, tc, T2,  x0=None):
        super().__init__(sigma, tc, x0)
        self.T2 = T2
        self.sig_fast = 1/T2*1e3

    def update(self, dt):
        return super().update(dt) + self.sig_fast * np.random.normal()
        

class Telegraph_Noise(NoiseProcess):
    def __init__(self, sigma, gamma, x0=0, state0 = None):
        self.gamma = gamma
        self.sigma = sigma
        self.x0 = x0
        if state0 is None:
            self.x = x0+ self.sigma*(2*np.random.randint(0, 2)-1)
        else:
            self.x = x0+self.sigma*state0


    def update(self, dt):
        # update telegraph noise
        switch_probability = 1/2 - 1/2*np.exp(-2*self.gamma*dt) 

        if np.random.uniform(0,1) < switch_probability:
            self.x = -self.x    
        return self.x + self.x0
    
    def set_state(self, state):
        self.x = self.x0 + self.sigma*state
        return self.x
    

class Over_f_noise(NoiseProcess):
    def __init__(self, n_telegraphs, S1 ,couplings_dispersion, ommax, ommin, om0=0, state = None):
        self.n_telegraphs = n_telegraphs
        self.S1 = S1
        self.state = state
        self.couplings_dispersion = couplings_dispersion
        self.ommax = ommax
        self.ommin = ommin
        self.sigma = S1*np.sqrt(np.log(ommax/ommin)*2)
        self.spawn_telegraphs(n_telegraphs, couplings_dispersion, state)
        self.om0 = om0
        self.x = om0 + np.sum([telegraph.x for telegraph in self.telegraphs])
       
        
    def get_sigma(self):
        return self.sigma
    
    def get_tc(self):
        return 1e-1
    

    def spawn_telegraphs(self, n_telegraphs, couplings_dispersion, state):
        if state is None:
            state = np.random.choice([-1,1], size = n_telegraphs)
        
        
        uni = np.random.uniform(0,1,size = n_telegraphs)
        gammas = self.ommax*np.exp(-np.log(self.ommax/self.ommin)*uni)
        
        sigmas_avg = self.S1 / np.sqrt(self.n_telegraphs / 2 / np.log(self.ommax / self.ommin))
        sigmas = sigmas_avg*np.random.normal(1,couplings_dispersion, size = n_telegraphs)
        self.telegraphs = []

        for n, gamma in enumerate(gammas):
            self.telegraphs.append(Telegraph_Noise
                                   (sigmas[n], gamma, state0 = state[n]))
        
    def update(self, dt):
        x = 0
        for telegraph in self.telegraphs:
            x += telegraph.update(dt)
        self.x = x + self.om0
        return self.x
    

        
    def reset(self):
        for n,telegraph in enumerate(self.telegraphs):
            telegraph.set_state(self.state[n])
        self.x = self.om0 + np.sum([telegraph.x for telegraph in self.telegraphs])
    
 

