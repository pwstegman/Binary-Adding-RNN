import os.path

if os.path.isfile("net.xml") or os.path.isfile("stats.p"):
	print("Network and/or stats file already exist, exiting")
	exit()

from pybrain.structure import RecurrentNetwork
from pybrain import LinearLayer, SigmoidLayer, FullConnection
from pybrain.tools.customxml.networkwriter import NetworkWriter
import pickle

n = RecurrentNetwork()
n.addInputModule(LinearLayer(2, name='in'))
n.addModule(SigmoidLayer(4, name='hidden'))
n.addOutputModule(SigmoidLayer(1, name='out'))
n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))
n.sortModules()

NetworkWriter.writeToFile(n, 'net.xml')
pickle.dump({"epochs":0, "time":0}, open( "stats.p", "wb" ) )
