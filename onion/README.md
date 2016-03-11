## MSM class with immutable objects


### DiscreteModel

A `Model` is the base class for all time homogeneous models on a discrete state space. Examples for this are `MSM`, `OOM`, `HMM`

```py
def path_probability(self, path):
def stationary_distribution(self):
def dt(self):
def mfpt(self):
def is_time_reversible(self):
	"""
	Returns whether pi(x1) * P([x1, ..., xN]) = pi(xN) * P([xN, ..., x1])
	"""
def connectivity(self):
def is_ergodic(self)
```

### BayesianDiscreteModel

Since all of these are estimated from data there exist Bayesian versions that represent all possible models of a certain size and implement a likelihood function that assigns each model a probability given the used data to parametrize it `BayesianMSM`, `BayesianOOM`, `BayesianHMM`

### MSM object

The MSM object is really just representing a single MSM. It provides all the properties you can estimate from a MSM

```py
class MSM(object):
	def __init__(self, P):
		"""
		Parameters
		----------
		P : numpy.ndarray, shape = (N, N)
			the representation of the transition matrix
		"""
		self.P = P
		
	def mfpt(self):
	def 
```

### StateSpace

Contains a definition what a finite set of states means. Derivatives could be `1D`, `2D`, `Voronoi`, `Arbitrary`, `Integer` / `Identity`, `FuzzyClustering`

$$ i : \mathbb{N} \mapsto S_i \subset \Omega $$

### DiscreteTrajectory object

Contains a single discretized trajectory and has a StateSpace to explain what the actual numbers mean.
Default will be `IdentityStates`

### MSMEstimator

Estimates a `BayesianMSM` from a `DiscreteTrajectory` object given specific settings

```py
class MSMEstimator(object):
	def __init__(self, lagtime):
		self.lagtime = lagtime
		
	def estimate(self, obj):
		"""
		Returns
		-------
		`BayesianMSM`
		"""

```

Variants could be `ReversibleEstimator`, `BoxerReversible`, ...

### BayesianMSM

The `BayesianMSM` represents a distribution of MSMs. Picking a particular MSM will return an `MSM` object

It has a function that returns a likelihood for a potential MSM (that is compatible in size and maybe the StateDefinition)

It can return the most likeli `MSM` object and return error estimations, variances etc...

You can also sample form the distribution of MSMs.

### Observable

A Mapping from the StateDefinition to a number

### Projection

A Mapping from a state space to state space

```py
class Projection(object):

```

### OOM

An `OOM` object is similar to `MSM` and will allow to predict similar quantities but has a different internal representation.

### BayesianHMM

