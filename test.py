from pybrain.structure import RecurrentNetwork
from pybrain import LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
import pickle

from random import randint

n = NetworkReader.readFrom('net.xml')
stats = pickle.load( open( "stats.p", "rb" ) )
epochs = stats["epochs"]
totaltime = stats["time"]

print("Loaded network from net.xml.", epochs, "epochs have already been run.", round(round(totaltime)/6)/10, "minutes spent training.")

tests = 100000
maxguess = 100000000
goods = 0
bads = 0

print("Running",tests,"tests")

for i in range(tests):

	if i % 1000 == 0:
		print(i/tests)

	an = randint(0,maxguess)
	bn = randint(0,maxguess)
	a = [int(x) for x in bin(an)[2:]]
	b = [int(x) for x in bin(bn)[2:]]
	r = [int(x) for x in bin(an+bn)[2:]]
	while len(a) < len(r):
		a = [0] + a

	while len(b) < len(r):
		b = [0] + b

	a = a[::-1]
	b = b[::-1]
	r = r[::-1]

	rn = []

	n.reset()
	for i in range(len(a)):
		inl = [a[i], b[i]]
		rn.append(str(int(round(n.activate(inl)[0]))))
	rn = rn[::-1]
	r = r[::-1]
	rint = int("".join(rn), 2)
	if rint == an+bn:
		goods += 1
	else:
		bads += 1
		print("Failed on", an, "+", bn, "=", rint, "instead of", an+bn)

print("Successful", goods, "Fails", bads)
