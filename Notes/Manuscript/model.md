# Model
## Model of 1/f noise
We model 1/f noise as a sum of multiple Ornstein-Uhlenmback process of the switching rate $\gamma$ drawn from the log-normal distribution. The spectrum of resulting noise can be written as:
$$
    S(\omega) = \sum_n \frac{2\sigma_n^2\gamma_n}{\gamma_n^2 + \omega^2} \approx \int \text{d}\gamma \, p(\gamma) \,\frac{2\sigma^2\gamma}{\gamma^2 + \omega^2} = \frac{A}{\omega},
$$
in which $p(\gamma) = 1/\gamma$ represents the distribution of the switching rates.

## Conditional probability and correlation function 


## Bayesian estimaton


## Characteristic frequencies

There are several natural frequencies appearing in the system. Starting from the highest we have:
- The inverse of single shot time $f_s = 1/\tau$. Above this frequency the noise is effectively white and can be treated with master equation. 
- The inverse of the estimation time $f_e = 1/\tau_{\text{est}}$. Noise below that frequency can be in principle estimated. 
- The inverse of the operation time $f_o = 1/\tau_op$, also the inverse of the time period between two consequtive estimations. The lowest frequency that contribute to qubit dephasing. 


