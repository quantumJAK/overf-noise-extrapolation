# Description: This file contains functions that simulate the dynamics of the qubit
import numpy as np
import constants_and_units as cu
import matplotlib.pyplot as plt

class Model:
    """
    Main class that contains functions that simulate the dynamics of the qubit

    Functions:
    -------------
    track_down -- tracks down the omega field, using selected estimation method (method).
    get_single_shot -- simulates a single shot of the estimation protocol
    measure -- simulates the projective measurment of the qubit
    run_single_realisation -- run single realisation of the estimation protocol
    did_T_relax -- simulates the relaxation of the qubit
    simulate_overlapped_gaussians -- simulates the measurement noise
    get_confidence -- calculates confidence of the measurment
    """

    def __init__(self, params):
        self.params = params

    def track_down(self, methods, noise):
        """
        This function tracks down the omega field, using selected estimation method (method).
        The field evolves according to noise model (noise).

        Parameters:
        -------------
        methods : list of instances of Estimation_method class
            used estimation method, e.g. Estimate_shulman()
        noise : instance of Noise class
            used nosie model of the frequency drift, e.g. OU_process()
        cycle_time : float
            qubit cycle time approx measurment time
        N_realisations : int
            number of realisations
        dt_realisations : float
            time between realisations
        Returns:
        -------------
        data : list
            data -- list of realisations r
            data[r] -- list of shots in rth realisation
            data[r][s] -- list of [shot number, time, bit, tau, om, pdf]
        pdfs : array[realisation][len(freqs)]
            array of posterior pdffs
        tracker : list
            list of tuples (timestamp, om)
        """
        noise.reset()
        for method in methods:
            method.init_pdf()

        data = []
        tracker = []
        pdfs = []
        time = 0
        for n_r in range(self.params["N_realisations"]):
            if n_r > 0:
                for method in methods:
                    method.reset_pdf(last_entry[1] + self.params["dt_realisations"], noise)
                   
            data.append(self.run_single_realisation(methods, noise))

            noise.update(self.params["dt_realisations"])
            # This part is ugly, but it is needed to track down the field with timestamps
            last_entry = data[-1][-1][-1]
            time += last_entry[1]  + self.params["dt_realisations"]

            tracker.append(
                [time, noise.get_value()] + [method.get_avg() for method in methods]
            )
            # The end of ugly part

            pdfs.append([method.get_pdf(method.freqs) for method in methods])
        return data, pdfs, tracker

    # to push

    def get_single_shot(self, method, noise, n):
        # This function simulates a single shot of the estimation protocol
        """
        Parameters:
        -------------
        method : instance of Estimation_method class    .
            used estimation method, e.g. Estimate_shulman()
        noise : instance of Noise class
            used nosie model of the frequency drift, e.g. OU_process()
        pdf :array
            current probability distribution or its moments
        cycle_time : float
            qubit cycle time approx measurment time
        n : int
            shot number in a given realisation
        """
        tau = method.next_tau(n)
        phi = self.integrate_evolution(noise, tau*cu.f2omGHz)
        bit, conf = self.measure(phi)
        # updates
        om = noise.update(self.params["cycle_time"])
        method.update_pdf(bit, tau, conf)


        return bit, tau, om

    def integrate_evolution(self, noise, tau, time_steps=10):
        """
        Integrate the evolution of the qubit for time tau, where the frequency is given by noise,
        using the trapezoidal rule for numerical integration.

        Parameters:
        -----------
        noise : instance of Noise class
            Noise model of the frequency drift, e.g. OU_process().
        tau : float 
            Time of the evolution.
        time_steps : int, optional
            Number of time steps used for numerical integration. Default is 10.

        Returns:
        --------
        phi : float
            Final phase of the qubit after the evolution for time tau.
        """
        dt = tau / time_steps
        t = np.linspace(0, tau, time_steps + 1)
        f = np.array([noise.update(dt) for i in range(time_steps + 1)])
        phi = np.trapz(f, t)
        return phi
    


    def measure(self, phi):
        """
        This function simulates the projective measurment of the qubit
        Parameters:
        -------------
        phi -- the coherenct phase of the qubit, either om*tau (quasi-static) or phi = \int om dt (fast noise)
        Returns:
        -------------
        bit -- result of the measurment (0 or 1)
        """

        state = {"S": 1, "T": 0}  # For easier reading S->1, T->0

        prob_singlet = 1 / 2 + 1 / 2 * np.cos(phi)

        bit = np.random.choice(
            [state["S"], state["T"]], p=[prob_singlet, 1 - prob_singlet]
        )
        confidence = 1
        

        if bit == state["T"]:
            bit = self.did_T_relax(bit, self.params["T1"])
        """
        bit, confidence = self.simulate_overlapped_gaussians(
            bit, width=1, separation=self.params["separation/width"]
        )
        """
        return bit, confidence

    def run_single_realisation(self, methods, noise):
        """
        Function that run single realisation of the estimation protocol
        Parameters:
        -------------
        methods : list of instances of Estimation_method class    .
            used estimation method, e.g. Estimate_shulman()
        noise : instance of Noise class
            used nosie model of the frequency drift, e.g. OU_process()
        cycle_time : float
            qubit cycle time approx measurment time
        pdf -- apriori probability distribution or its moments
        Returns:
        -------------
        data_shots : list
            data_shots -- list of shots in the realisation
            data_shots[s] -- list of [shot number, time, bit, tau, om, pdf]
        """

        get_shots = True
        data_shots = []
        for m in methods:
            data_shots.append([])

        timestamp = 0
        
        for m, method in enumerate(methods):
            n = -1  # quick fix TODO: fix this
            get_shots = True
            om = noise.get_value()
            data_shots[m].append([n, timestamp, None, None, om, method.get_pdf(method.freqs)]) #TODO FIX THAT
            n += 1
            while get_shots:
                bit, tau, om = self.get_single_shot(method, noise, n)
                timestamp += tau
                data_shots[m].append([n, timestamp, bit, tau, om, method.get_pdf(method.freqs)]) #TODO FIX THAT
                n += 1
                get_shots = method.if_terminate(n, timestamp)
                timestamp += self.params["cycle_time"]
                
        return data_shots

    def did_T_relax(self, bit, T1):
        # TODO: documentation
        prob_relax = 1 - np.exp(-self.params["cycle_time"] / T1)
        bit = np.random.choice([1, 0], p=[prob_relax, 1 - prob_relax])
        return bit

    def simulate_overlapped_gaussians(self, bit, separation, width):
        # TODO: documentation
        # Separaton and width can be passed as params["separation"] and params["width"], which one is better?
        if bit == 0:
            I = np.random.normal(loc=-separation, scale=width)
        else:
            I = np.random.normal(loc=separation / 2, scale=width)

        confidence = self.get_confidence(I, separation, width)
        return I > 0, confidence

    def get_confidence(self, I, separation, width):
        """
        Function that calculates confidence of the measurment
        #TODO: implement less aribtrary, more physically/mathematically motivated function
        0 -> no confidence, 1 -> full confidence
        Parameters:
        -------------
        I : float
            measured current
        separation : float
            separation of the gaussians
        width : float
            width of the gaussians
        Returns:
        -------------
        confidence : float
            confidence of the measurment
        """
        confidence = np.abs(2 * (1 / (1 + np.exp(-I * separation / width**2))) - 1)
        return confidence
