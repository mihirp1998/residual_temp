import pickle
from collections import defaultdict

d = defaultdict(lambda: [])
a = pickle.load(open("/Users/mihir/Documents/projects/kinetic_data/kinetic/tempValidHyperTuple10.p","rb"))
for i in a:
	d[i[0].split("/")[1][:-9]].append(i[0])
print(d['frameAnNDDIUSuxM_000027_000037'])