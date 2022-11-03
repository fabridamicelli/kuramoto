[![Downloads](https://static.pepy.tech/personalized-badge/kuramoto?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/kuramoto)

# kuramoto
Python implementation of the Kuramoto model.

## Install
```bash
pip install kuramoto

```

## Usage
```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from kuramoto import Kuramoto, plot_phase_coherence, plot_activity

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.6)

# Interactions are represented as an adjacency matrix _A_, a 2D numpy ndarray.
# Instantiate a random graph and transform into an adjacency matrix
graph_nx = nx.erdos_renyi_graph(n=100, p=1) # p=1 -> all-to-all connectivity
graph = nx.to_numpy_array(graph_nx)

# Instantiate model with parameters
model = Kuramoto(coupling=3, dt=0.01, T=10, n_nodes=len(graph))

# Run simulation - output is time series for all nodes (node vs time)
act_mat = model.run(adj_mat=graph)

# Plot all the time series
plot_activity(act_mat)
```
![png](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/timeseries.png)

```python
# Plot evolution of global order parameter R_t
plot_phase_coherence(act_mat)
```
![png](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/orderparam.png)
       
```python
# Plot oscillators in complex plane at times t = 0, 250, 500
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5),
                         subplot_kw={
                             "ylim": (-1.1, 1.1),
                             "xlim": (-1.1, 1.1),
                             "xlabel": r'$\cos(\theta)$',
                             "ylabel": r'$\sin(\theta)$',
                         })

times = [0, 200, 500]
for ax, time in zip(axes, times):
    ax.plot(np.cos(act_mat[:, time]),
            np.sin(act_mat[:, time]),
            'o',
            markersize=10)
    ax.set_title(f'Time = {time}')
```
![png](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/oscillators.png)

As a sanity check, let's look at the phase transition of the global order parameter (_R<sub>t_) as a function of coupling (_K_) (find critical coupling _K<sub>c_) and compare with numerical results already published by English, 2008 (see Ref.) – we will match those model parameters.

```python
# Instantiate a random graph and transform into an adjacency matrix
n_nodes = 500
graph_nx = nx.erdos_renyi_graph(n=n_nodes, p=1) # p=1 -> all-to-all connectivity
graph = nx.to_numpy_array(graph_nx)

# Run model with different coupling (K) parameters
coupling_vals = np.linspace(0, 0.6, 100)
runs = []
for coupling in coupling_vals:
    model = Kuramoto(coupling=coupling, dt=0.1, T=500, n_nodes=n_nodes)
    model.natfreqs = np.random.normal(1, 0.1, size=n_nodes)  # reset natural frequencies
    act_mat = model.run(adj_mat=graph)
    runs.append(act_mat)

# Check that natural frequencies are correct (we need them for prediction of Kc)
plt.figure()
plt.hist(model.natfreqs)
plt.xlabel('natural frequency')
plt.ylabel('count')
```
![png](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/nat_freq_dist.png)


```python
# Plot all time series for all coupling values (color coded)
runs_array = np.array(runs)

plt.figure()
for i, coupling in enumerate(coupling_vals):
    plt.plot(
        [model.phase_coherence(vec)
         for vec in runs_array[i, ::].T],
        c=str(1-coupling),  # higher -> darker   
    )
plt.ylabel(r'order parameter ($R_t$)')
plt.xlabel('time')
```
![png](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/ts_diff_K.png)


```python
# Plot final Rt for each coupling value
plt.figure()
for i, coupling in enumerate(coupling_vals):
    r_mean = np.mean([model.phase_coherence(vec)
                      for vec in runs_array[i, :, -1000:].T]) # mean over last 1000 steps
    plt.scatter(coupling, r_mean, c='steelblue', s=20, alpha=0.7)

# Predicted Kc – analytical result (from paper)
Kc = np.sqrt(8 / np.pi) * np.std(model.natfreqs) # analytical result (from paper)
plt.vlines(Kc, 0, 1, linestyles='--', color='orange', label='analytical prediction')

plt.legend()
plt.grid(linestyle='--', alpha=0.8)
plt.ylabel('order parameter (R)')
plt.xlabel('coupling (K)')
sns.despine()
```
![png](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/phase_transition.png)


## Kuramoto model 101
- The [Kuramoto model](https://en.wikipedia.org/wiki/Kuramoto_model) is used to study a wide range of systems with synchronization behaviour.
- It is a system of _N_ coupled periodic oscillators.
- Each oscillator has its own _natural frequency_ _omega<sub>i_, i.e., constant angular velocity. 
- Usually, the distribution of natural frequencies is choosen to be a gaussian-like symmetric function.
- A random initial (angular) position _theta<sub>i_ is assigned to each oscillator.
- The oscillator's state (position) _theta<sub>i_ is governed by the following differential equation:

![jpg](https://github.com/fabridamicelli/kuramoto_model/blob/master/images/derivative.jpg)
      

where K is the coupling parameter and _M<sub>i_ is the number of oscillators interacting with oscillator _i_. 
_A_ is the _adjacency matrix_ enconding the interactions - typically binary and undirected (symmetric), such that if node _i_ interacts with node _j_, _A<sub>ij_ = 1, otherwise 0.
The basic idea is that, given two oscillators, the one running ahead is encouraged to slow down while the one running behind to accelerate.

In particular, the classical set up has _M = N_, since the interactions are all-to-all (i.e., a complete graph). Otherwise, _M<sub>i_ is the degree of node _i_.

## Kuramoto model 201
A couple of facts in order to gain intuition about the model's behaviour:
- If synchronization occurs, it happens abruptly.
- That is, synchronization might not occur.
- Partial synchronization is a possible outcome.
- The order parameter _R<sub>t_ measures global synchronization at time _t_. It is basically the normalized length of the sum of all vectors (oscillators in the complex plane).
- About the global order parameter _R<sub>t_:
  - constant, in the double limit N -> inf, t -> inf
  - independent of the initial conditions
  - depends on coupling strength
  - it shows a sharp phase transition (as function of coupling)
- Steady solutions can be computed assuming _R<sub>t_ constant. The result is basically that each oscillator responds to the mean field produced by the rest.
- In the all-to-all connected scenaria, the critical coupling _K<sub>c_ can be analytically computed and it depends on the spread of the natural frequencies distribution (see English, 2008)
- The higher the dimension of the lattice on which the oscillators are embedded, the easier it is to synchronize. For example, there isn't any good synchronization in one dimension, even with strong coupling. In two dimensions it is not clear yet. From 3 dimensions on, the model starts behaving more like the mean field prediction.

For more and better details, [this talk](https://www.youtube.com/watch?v=5zFDMyQ8z8g) by the great Steven Strogatz is a nice primer.

## Requirements
- numpy
- scipy
- matplotlib
- For the examples:
  - bctpy
  - networkx
  - seaborn

## Tests
Run tests with
```bash
make test
```

## Citing

If you find this package useful for a publication, then please use the following BibTeX to cite it:
```
@misc{kuramoto,
  author = {Damicelli, Fabrizio},
  title = {Python implementation of the Kuramoto model},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fabridamicelli/kuramoto}},
}
```

## References & links 
- [English, 2008](http://doi.org/10.1088/0143-0807/29/1/015). Synchronization of oscillators: an ideal introduction to phase transitions.
- [Dirk Brockmann's explorable](http://www.complexity-explorables.org/explorables/kuramoto/). “Ride my Kuramotocycle!”.
- [Math Insight - applet](https://mathinsight.org/applet/kuramoto_order_parameters). The Kuramoto order parameters.
- [Kuramoto, Y. (1984)](http://doi.org/10.1007/978-3-642-69689-3). Chemical Oscillations, Waves, and Turbulence.
- [Steven Strogatz - talk](https://www.youtube.com/watch?v=5zFDMyQ8z8g). Coupled Oscillators That Synchronize Themselves
