import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, bernoulli

##Part A
stepSize = 0.05
thetaVector = np.array([i/100 for i in range(0, 100+int(stepSize*100), int(stepSize*100))])

##Part B
fig, axes = plt.subplots(1, 1)
axes.plot(thetaVector, beta.pdf(thetaVector, 21, 11), 'k-', lw=2, label='beta(21,11)')
axes.legend(loc='best', frameon=False)
plt.title('Prior Distribution')
plt.xlabel('Theta values')
plt.ylabel('Frequency')
plt.show()

##Part C
def simFairFlips(n):
	heads = 0
	for i in range(n):
		heads += random.randint(0, 1)
	return heads
heads = simFairFlips(128)
plt.bar([0, 0.5], [heads, 128-heads], align='center', width=0.3)
plt.xticks([0, 0.5], ['Heads', 'Tails'])
plt.title('128 Simulated Fair Coin Flips')
plt.show()

##Part D
def likelihoodFunction(theta, T, s):
	return (theta**s) * ((1-theta)**(T-s))

#Likelihoods is a dictionary of sample sizes, each key containing a vector, likelihoodFunction(theta), which is applied to each value in thetaVector [0, 0.05, ..., 1]
likelihoods = {1:[], 2:[], 4:[], 8:[], 16:[], 32:[], 64:[], 128:[]}
for n in [1, 2, 4, 8, 16, 32, 64, 128]:
	successes = simFairFlips(n)
	for theta in thetaVector:
		l = likelihoodFunction(theta, n, successes)
		likelihoods[n] += [l]

##Part E
posterior = {1:[], 2:[], 4:[], 8:[], 16:[], 32:[], 64:[], 128:[]}
for n in posterior.keys():
	posterior[n] = beta.pdf(thetaVector, 21, 11) * likelihoods[n]
	s = sum(thetaVector)
	posterior[n] = [i/s/stepSize for i in posterior[n]]
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex=False, sharey=False)
# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex=True, sharey=True)
ax1.plot(thetaVector, posterior[1], 'k-', lw=2)
ax1.set_title('1 Coin Flip')
ax2.plot(thetaVector, posterior[2], 'k-', lw=2)
ax2.set_title('2 Coin Flips')
ax3.plot(thetaVector, posterior[4], 'k-', lw=2)
ax3.set_title('4 Coin Flips')
ax4.plot(thetaVector, posterior[8], 'k-', lw=2)
ax4.set_title('8 Coin Flips')
ax5.plot(thetaVector, posterior[16], 'k-', lw=2)
ax5.set_title('16 Coin Flips')
ax6.plot(thetaVector, posterior[32], 'k-', lw=2)
ax6.set_title('32 Coin Flips')
ax7.plot(thetaVector, posterior[64], 'k-', lw=2)
ax7.set_title('64 Coin Flips')
ax8.plot(thetaVector, posterior[128], 'k-', lw=2)
ax8.set_title('128 Coin Flips')
fig.suptitle('Posterior Distributions by Sample Size', fontsize=30)
fig.text(0.5, 0.04, 'theta', ha='center', va='center', fontsize=20)
fig.text(0.06, 0.5, 'P(theta)', ha='center', va='center', rotation='vertical', fontsize=20)
plt.show()

# Alternative Layout
# fig, axes = plt.subplots(1, 1)
# alpha = simFairFlips(128)
# axes.plot(thetaVector, beta.pdf(thetaVector, alpha, 128-alpha), 'k-', lw=2, label='128 Simulated flips')
# axes.legend(loc='best', frameon=False)
# plt.xlabel('Theta values')
# plt.ylabel('Frequency')
# plt.show()
