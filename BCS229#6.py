import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm


def getSample(n, p):
	random.seed()
	return [1 if random.random() <= p else 0 for i in range(n)]

def sampleTimepoints(n, p):
	random.seed()
	return sum([1 for i in range(n) if random.random() <= p])

def poissonSampleTimepoints(n, p):
	return sum(poisson.rvs(p, size=n))

def fanoFactor(data):
	data = np.array(data)
	return data.var()/data.mean()


#Part A
histo = [sampleTimepoints(1000, 0.0032) for i in range(10000)]
print("Fano factor: "+str(fanoFactor(histo)))

fig, axes = plt.subplots(1, 1)
axes.hist(histo, color='green')
axes.set_title('10k samples of Poisson process, mean=3.2')
plt.xlabel('Spikes per sample')
plt.ylabel('Frequency')
plt.show()

#Part B
histo = [sampleTimepoints(1000, 0.0095) for i in range(10000)]
print("Fano factor: "+str(fanoFactor(histo)))

fig, axes = plt.subplots(1, 1)
axes.hist(histo, color='green')
axes.set_title('10k samples of Poisson process, mean=9.5')
plt.xlabel('Spikes per sample')
plt.ylabel('Frequency')
plt.show()

#Part C
histo = [poissonSampleTimepoints(1000, 0.0032) for i in range(10000)]
print("Distribution fano factor: "+str(fanoFactor(histo)))

fig, axes = plt.subplots(1, 1)
axes.hist(histo, color='blue')
axes.set_title('10k samples of real Poisson process, mean=3.2')
plt.xlabel('Spikes per sample')
plt.ylabel('Frequency')
plt.show()

histo = [poissonSampleTimepoints(1000, 0.0095) for i in range(10000)]
print("Distribution fano factor: "+str(fanoFactor(histo)))

fig, axes = plt.subplots(1, 1)
axes.hist(histo, color='blue')
axes.set_title('10k samples of real Poisson process, mean=9.5')
plt.xlabel('Spikes per sample')
plt.ylabel('Frequency')
plt.show()

#Part D
histo = []
for i in range(10000):
	s = getSample(1000, 0.0032)
	diffs = np.diff(np.array([i for i in range(len(s)) if s[i] == 1]))
	histo += diffs.tolist()
fig, axes = plt.subplots(1, 1)
axes.hist(histo, 100, color='purple')
axes.set_title('10k samples of spike intervals, mean=3.2')
plt.xlabel('Time interval between spikes')
plt.ylabel('Interval frequency')
plt.show()

histo = []
for i in range(10000):
	s = getSample(1000, 0.0095)
	diffs = np.diff(np.array([i for i in range(len(s)) if s[i] == 1]))
	histo += diffs.tolist()
fig, axes = plt.subplots(1, 1)
axes.hist(histo, 100, color='purple')
axes.set_title('10k samples of spike intervals, mean=9.5')
plt.xlabel('Time interval between spikes')
plt.ylabel('Interval frequency')
plt.show()

#Part 2
neuronSpacing = 1
stimulus = 0
sigmatc = 20
N = 1000
samples = 150

gain = 1
s = np.arange(-N*neuronSpacing/2+stimulus, N*neuronSpacing/2+stimulus+neuronSpacing, neuronSpacing)
neuronPopulation = [norm(spref, sigmatc**2) for spref in s] #Tile gaussian neurons along stimuli
neuronProbabilities = [gain*neuron.pdf(stimulus) for neuron in neuronPopulation] #Get population's response rates for a given stimulus
MLerror = []
likelihoodVar = []
for r in range(samples):
	spikeSample = [poissonSampleTimepoints(10000, p) for p in neuronProbabilities] #Poisson sample the population
	spikeSum = sum(spikeSample)
	MLest = np.dot(spikeSample, s)/spikeSum
	MLerror += [(stimulus - MLest) ** 2]
	likelihoodVar += [sigmatc/np.sqrt(spikeSum)]
	print(str(r)+": "+str(MLerror[-1]))

print(np.corrcoef(MLerror, likelihoodVar))
fig, axes = plt.subplots(1, 1)
axes.scatter(MLerror, likelihoodVar)
plt.show()

gain = 10
s = np.arange(-N*neuronSpacing/2+stimulus, N*neuronSpacing/2+stimulus+neuronSpacing, neuronSpacing)
neuronPopulation = [norm(spref, sigmatc**2) for spref in s] #Tile gaussian neurons along stimuli
neuronProbabilities = [gain*neuron.pdf(stimulus) for neuron in neuronPopulation] #Get population's response rates for a given stimulus
MLerror = []
likelihoodVar = []
for r in range(samples):
	spikeSample = [poissonSampleTimepoints(10000, p) for p in neuronProbabilities] #Poisson sample the population
	spikeSum = sum(spikeSample)
	MLest = np.dot(spikeSample, s)/spikeSum
	MLerror += [(stimulus - MLest) ** 2]
	likelihoodVar += [sigmatc/np.sqrt(spikeSum)]
	print(str(r)+": "+str(MLerror[-1]))

print(np.corrcoef(MLerror, likelihoodVar))
fig, axes = plt.subplots(1, 1)
axes.scatter(MLerror, likelihoodVar)
plt.show()