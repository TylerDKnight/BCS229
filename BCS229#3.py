import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

#Question 2
##Part A
stepSize = 0.2
svals = np.array([i/10 for i in range(0, 400+1*int(stepSize*10), int(stepSize*10))])

##Part B, C, D, E
priorDist = norm(20, 4)
likelihoodDist = norm(30, 5)
prior = []
likelihood = []
for s in svals:
	prior += [priorDist.pdf(s)]
	likelihood += [likelihoodDist.pdf(s)]
prior = np.array(prior)
likelihood = np.array(likelihood)
np.set_printoptions(precision=3, suppress=True)

posterior = np.array(np.array(prior) * np.array(likelihood))
posterior = posterior / sum(posterior) / stepSize
print(np.ndarray.std(prior))
print(np.ndarray.std(likelihood))
print(np.ndarray.std(posterior))

fig, axes = plt.subplots(1, 1)
axes.plot(svals, posterior, 'k-', lw=3, label='Posterior percept distribution')
axes.plot(svals, prior, 'r-.', lw=3, alpha=0.7, label='Stimulus distribution')
axes.plot(svals, likelihood, 'b-.', lw=3, alpha=0.7, label='Measurement distribution')
axes.legend(loc='best', frameon=False)
plt.show()

#Question 3
stimulusDist = norm(20, 4) #p(s)
measurementDist = norm(10, 5) #since s=10, p(x|s) => p(x|10)
measurement = [measurementDist.pdf(x) for x in svals]
prior = [stimulusDist.pdf(x) for x in svals]
fig, axes = plt.subplots(1, 1)
for i in range(10):
	sample = measurementDist.rvs() #randomly sample from p(x|10), the measurement distribution
	plt.axvline(x=sample, alpha=0.2) #Vertically mark sample location on graph
	likelihood = []
	for s in svals:
		likelihood += [norm(sample, 5).pdf(s)]
	posterior = np.array(prior) * np.array(likelihood)
	posterior = posterior / sum(posterior) / stepSize
	if i == 1:
		axes.plot(svals, posterior, 'k-', alpha=0.5, lw=2, label='Sample posterior')
		axes.plot(svals, likelihood, 'b.', alpha=0.35, lw=1, label='Sample measurement')
	else:
		axes.plot(svals, posterior, 'k-', alpha=0.5, lw=2)
		axes.plot(svals, likelihood, 'b.', alpha=0.35, lw=1)
axes.plot(svals, measurement, 'g-.', lw=5, label='Measurement distribution')
axes.plot(svals, prior, 'r-.', lw=5, label='Stimulus Distribution')
axes.legend(loc='best', frameon=False)
plt.show()


#Question 4
stimulusDist = norm(20, 4) #p(s)
trueStimuli = []
MAPestimates = []
MLE = []
for i in range(100):
	stimulusSample = stimulusDist.rvs() #Drawing an s
	trueStimuli += [stimulusSample]
	measurementDist = norm(stimulusSample, 5) #p(x|sample)
	sample = measurementDist.rvs()
	prior = []
	likelihood = []
	for s in svals:
		prior += [stimulusDist.pdf(s)]
		likelihood += [norm(sample, 5).pdf(s)]
	posterior = np.array(prior) * np.array(likelihood)
	posterior = posterior / sum(posterior) / stepSize
	MAPestimates += [svals[posterior.tolist().index(max(posterior))]]
	MLE += [svals[likelihood.index(max(likelihood))]]

print(np.array(trueStimuli))
print(np.array(MAPestimates))
print(MLE)
fig, (axes1, axes2) = plt.subplots(2, 1)
axes1.set_title('Maximum-a-posteriori Estimation')
axes1.plot(sorted(trueStimuli), sorted(MAPestimates), 'b-', lw=3)
axes2.set_title('Maximum Likelihood Estimation')
axes2.plot(sorted(trueStimuli), sorted(MLE), 'g-', lw=3)
plt.show()