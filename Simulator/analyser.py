import numpy as np
import constants_and_units as cu
import noise_models as noise_models
import simulation as simulation


def get_posterior_stats(data, methods, select_methods="all"):
    """
    This function computes the variance of the posterior pdfs
    Parameters
    -------------
    list : data
        structure: data[realisation][method][shot] = [timebin, bit, tau, om, pdf]
    methods : list of instances of EstimationMethod class
        list of methods [Shulman, Shulman_adaptive etc.]
    select_methods : list of ints
        list of indices of methods to be used. ex. [0,1] if only first and second method are to be used
    Returns
    -------------
    variances : array
        array of variances of the posterior pdfs (see structure below)
    errors: array
        array of errors, i.e. the difference between real and estimated om. (see structure below)
        structure: averages[realisations][method][shots] = avg
    """
    if select_methods == "all":
        select_methods = range(len(methods))

    # Should we write a faster version of this?
    vars = np.zeros((len(data), len(select_methods), len(data[0][0])))
    errors = np.zeros((len(data), len(select_methods), len(data[0][0])))
    for rn in range(len(vars)):
        for mn in range(len(vars[0])):
            for sn in range(len(vars[0][0])):
                vars[rn, mn, sn] = methods[mn].get_std(data[rn][mn][sn][-1]) ** 2
        
                errors[rn, mn, sn] = (
                    methods[mn].get_estimate(data[rn][mn][sn][-1])
                    - np.abs(data[rn][mn][sn][-2])
                )
    return vars, errors


def get_statistics_over_trials(methods, noise_type, simulator):
    
    params = simulator.params
    stats_var = np.zeros((len(methods), params["trials"], params["N_realisations"], params["N_shots"]+1))
    stats_err = np.zeros((len(methods), params["trials"], params["N_realisations"], params["N_shots"]+1))
    time_traj = np.linspace(0,2*params["N_shots"]*params["N_realisations"]*params["cycle_time"],1001)

    for trial in range(params["trials"]):
        print(str(100*trial/params["trials"])+"%")
        #Trajectory = noise_models.SingleTrajectory(time_traj, noise_type)
        for mn,method in enumerate(methods):

            data, pdfs, tracker = simulator.track_down(methods = [method], 
            noise = noise_type)  
            
            
            post_variances, errors = get_posterior_stats(data, 
                                                    methods = [method],
                                                    select_methods = "all")
            

            stats_var[mn,trial] = post_variances[:,0,:]
            stats_err[mn,trial] = errors[:,0,:]

    return stats_var, stats_err




# def get_taus(data, methods, select_methods = "all"):
#     if select_methods == "all":
#         select_methods = range(len(methods))

#     realisations = len(data)
#     shots = len(data[0][0])
#     taus = np.zeros((len(select_methods), realisations, shots))
#     for rn in range(realisations):
#         for sn in range(shots):
#             for mn0,mn in enumerate(select_methods):
#                 taus[mn0, rn, sn] = methods[mn].get_tau(data[rn][mn][sn][-1])
