import math
import matplotlib.pyplot as plt

def generateHypotheses(n, X, axis):
	hyps = []
	for x in X:
		for l in range(1, n+1):
			for i in range(0, l):
				h = list(range(x-i, x-i+l))
				if h not in hyps:
					hyps += [h]
	return hyps

def computePosterior(n, x, axis):
	if len(x) == 0:
		return [0]*len(axis)
	else:
		hspace = generateHypotheses(n, x, axis)
		density = []
		for y in axis:
			ysum = []
			for h in hspace:
				term = 1/(len(h)**len(x))
				if y in h and (False not in [i in h for i in x]):
					ysum += [term]
			density += [sum(ysum)]
		m = max(density)
		for i in range(len(density)):
			density[i] = density[i] / m
		return density

##########
#Part A
minV = 38
maxV = 52
axis = range(minV, maxV+1)
n = 6
x = [45]
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Single Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|x)')
plt.show()
##########

##########
##Part B
minV = 0
maxV = 100
axis = range(minV, maxV+1)
n = 40
x = [43,44,45]
h = generateHypotheses(n,x,axis)
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Multiple Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|X)')
plt.show()
##########

##########
##Part C
minV = 0
maxV = 100
axis = range(minV, maxV+1)
n = 40
x = [37,42,45]
h = generateHypotheses(n,x,axis)
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Multiple Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|X)')
plt.show()
##########

##########
##Part D
minV = 0
maxV = 100
axis = range(minV, maxV+1)
n = 40
x = [15,35,45]
h = generateHypotheses(n,x,axis)
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Multiple Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|X)')
plt.show()
##########

##########
##Part E
minV = 0
maxV = 100
axis = range(minV, maxV+1)
n = 40
x = [37,45]
h = generateHypotheses(n,x,axis)
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Multiple Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|X)')
plt.show()
##########

##########
##Part F
minV = 0
maxV = 100
axis = range(minV, maxV+1)
n = 40
x = [37,40,42,45]
h = generateHypotheses(n,x,axis)
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Multiple Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|X)')
plt.show()
##########

##########
##Part G
minV = 0
maxV = 100
axis = range(minV, maxV+1)
n = 40
x = [37,38,40,40,41,42,43,45]
h = generateHypotheses(n,x,axis)
density = computePosterior(n, x, axis)

fig, axes = plt.subplots(1, 1)
axes.plot(axis, density, color='green')
axes.set_title('Multiple Sample Distribution: '+str(x))
plt.xlabel('X')
plt.ylabel('P(y in C|X)')
plt.show()
##########