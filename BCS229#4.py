import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform


##Parts A-F
stepSize = 0.05
xMin = 0
xMax = 14
scalingFactor = int(np.power(10, np.ceil(np.log10(1/stepSize))))
svals = np.array([i/scalingFactor for i in range(xMin*scalingFactor, int(xMax+stepSize)*scalingFactor, int(stepSize*scalingFactor))])

priorDist = uniform(xMin, xMax)
aStimulusDist = norm(5, 2)
vStimulusDist = norm(10, 1)
aDrawMeasurement = aStimulusDist.rvs()
vDrawMeasurement = vStimulusDist.rvs()

aMeasurement = []
vMeasurement = []
prior = []
for s in svals:
	prior += [priorDist.pdf(s)]
	aMeasurement += [norm(aDrawMeasurement, 2).pdf(s)]
	vMeasurement += [norm(vDrawMeasurement, 1).pdf(s)]
aMeasurement = np.array(aMeasurement) * prior
vMeasurement = np.array(vMeasurement) * prior
combinedLikelihood = np.array(aMeasurement) * np.array(vMeasurement)
posterior = combinedLikelihood / sum(combinedLikelihood) / stepSize
measuredMAP = svals[posterior.tolist().index(max(posterior))]
computedMAP = (aDrawMeasurement/np.var(aMeasurement)+vDrawMeasurement/np.var(vMeasurement)) / (1/np.var(aMeasurement)+1/np.var(vMeasurement))

np.set_printoptions(precision=3, suppress=True)
fig, axes = plt.subplots(1, 1)
axes.plot(svals, aMeasurement, 'g.', lw=3, label='Auditory Measurement Distribution')
axes.plot(svals, vMeasurement, 'g.', lw=3, label='Visual Measurement Distribution')
axes.plot(svals, combinedLikelihood, 'k-.', lw=3, label='Combined Measurement Distribution')
axes.plot(svals, posterior, 'k-', lw=3, label='Posterior Combined Distribution')
axes.plot(svals, prior, 'r-.', lw=3, alpha=0.7, label='Prior Uniform Distribution')
axes.plot(svals, [aStimulusDist.pdf(x) for x in svals], 'b-.', lw=3, alpha=0.2, label='Auditory Stimulus Distribution')
axes.plot(svals, [vStimulusDist.pdf(x) for x in svals], 'b-.', lw=3, alpha=0.2, label='Visual Stimulus Distribution')
plt.axvline(x=measuredMAP, alpha=0.2, label='Measured MAP Posterior')
plt.axvline(x=computedMAP, alpha=0.2, label='Computed MAP Posterior')
axes.legend(loc='best', frameon=False)
axes.set_title('Combined Cues Sampled from Stimulus Distributions')
plt.xlabel('S values')
plt.ylabel('Probability')
plt.show()


fig, axes = plt.subplots(1, 1)
axes.plot(svals, combinedLikelihood, 'k-.', lw=3, label='Combined Measurement Distribution')
axes.legend(loc='best', frameon=False)
axes.set_title('Combined Likelihood Functions Close Up')
plt.xlabel('S values')
plt.ylabel('Probability')
plt.show()


##Parts G-J
MAPestimates = []
sampleSize = 200
for i in range(sampleSize):
	aDrawMeasurement = aStimulusDist.rvs()
	vDrawMeasurement = vStimulusDist.rvs()
	aMeasurement = []
	vMeasurement = []
	for s in svals:
		aMeasurement += [norm(aDrawMeasurement, 2).pdf(s)]
		vMeasurement += [norm(vDrawMeasurement, 1).pdf(s)]
	aMeasurement = np.array(aMeasurement) * prior
	vMeasurement = np.array(vMeasurement) * prior
	combinedLikelihood = np.array(aMeasurement) * np.array(vMeasurement)
	posterior = combinedLikelihood / sum(combinedLikelihood) / stepSize
	measuredMAP = svals[posterior.tolist().index(max(posterior))]
	MAPestimates += [measuredMAP]

wa = (1/aStimulusDist.var())/(1/aStimulusDist.var()+1/vStimulusDist.var())
wv = (1/vStimulusDist.var())/(1/aStimulusDist.var()+1/vStimulusDist.var())
sHat = wa*aStimulusDist.mean() + wv*vStimulusDist.mean()
RAB = (np.mean(MAPestimates) - aStimulusDist.mean()) / (vStimulusDist.mean() - aStimulusDist.mean())
print("MAP measurements mean: "+str(np.mean(MAPestimates)))
print("Predicted MAP mean: "+str(sHat))
print("Relative Auditory Bias: "+str(RAB))

fig, axes = plt.subplots(1, 1)
axes.hist(MAPestimates, 20, color='green')
axes.set_title('Distribution of MAP Estimates')
plt.xlabel('S values')
plt.ylabel('Frequency')
plt.show()
