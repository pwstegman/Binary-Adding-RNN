from pybrain.structure import RecurrentNetwork
from pybrain import LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
from time import time
import pickle

n = NetworkReader.readFrom('net.xml')
stats = pickle.load( open( "stats.p", "rb" ) )
epochs = stats["epochs"]
totaltime = stats["time"]

print("Loaded network from net.xml.", epochs, "epochs have already been run.", round(totaltime/36)/100, "hours spent training.")

ds = SequentialDataSet(2,1)

from random import randint

start = time()
for i in range(1000):
	an = randint(0, 1000000)
	bn = randint(0, 1000000)
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

	ds.newSequence()
	for i in range(len(a)):
		inl = [a[i], b[i]]
		out = [r[i]]
		ds.appendLinked(inl, out)

#trainer = RPropMinusTrainer(n, dataset = ds)
trainer = BackpropTrainer(n, dataset = ds)

print("Generating dataset took", time()-start)

lastlen = 0

start = time()

try:
	while True:
		epochstart = time()
		error = trainer.train()
		tpe = time()-epochstart
		epochs += 1
		out = str(error) + " error " + str(epochs) + " epochs " + str(tpe) + " time per epoch"
		clearspaces = " "*(lastlen-len(out))
		lastlen = len(out)
		if epochs%100 == 0:
			thisrun = time()-start
			totaltime += thisrun
			start = time()
			try:
				NetworkWriter.writeToFile(n, 'net.xml')
				pickle.dump({"epochs":epochs, "time":totaltime}, open( "stats.p", "wb" ) )
				print("Autosaved network and stats after 100 epochs and ",thisrun,"seconds. Total time",round(totaltime/36)/100, "hours")
			except:
				print("Error autosaving network")
		print(out)
except KeyboardInterrupt:
	pass

thisrun = time()-start
totaltime += thisrun

print()
print()

print("Time spent training",thisrun,"Total time spent training",totaltime)

print("Error", error)
NetworkWriter.writeToFile(n, 'net.xml')
print("Saved network state to net.xml")
pickle.dump({"epochs":epochs, "time":totaltime}, open( "stats.p", "wb" ) )
print("Saved epochs to epochs.p")

while True:
	an = int(input("in_1: "))
	bn = int(input("in_2: "))
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
	print("Calculated \t", rint,"\t", "".join(rn))
	print("Expected   \t", an+bn,"\t", "".join(map(str, r)))
	
