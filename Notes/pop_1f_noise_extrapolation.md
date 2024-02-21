# Purpose
In the following note we consider 1/f noise which has both high- and low- frequency compontents. The presence of both makes it difficult to directly apply typicaly used high-pass filters in form of dynamical decoupling (echo) and online Hamiltonian learning. On the other hand the teporally correlated noise (low-frequency) is not mitigated by the standard error mitigation techniques. We propose to connect both techniques and apply the zero-noise extrapolation for the signal that is already pre-processed using beyesian Hamiltonian learning. The hope is that the latter can remove the noise at the timescale slower than the estimation time, which for spin qubits is typially in the range of $100 \mu s$.



