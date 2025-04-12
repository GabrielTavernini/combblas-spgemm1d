#!/global/homes/y/yuxihong/miniconda3/bin/ python3
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import socket
import argparse

hostname = socket.gethostname()
print("Hostname:", hostname)
if hostname.startswith("chitu"):
    abspath = "/home/hongy0a/graphclustering/commana"
else:
    abspath = "/pscratch/sd/y/yuxihong/graphclustering/commana"


# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments
parser.add_argument('--perm', type=str,required=True)
parser.add_argument('--gentype',type=int, default=5,required=True)
parser.add_argument('--nprocs',type=int,required=True)
parser.add_argument('--dataset',type=str,required=True)

# Parse arguments
args = parser.parse_args()

# perm = "RandPerm"
# dataset = "cage15"
# gentype = 5
# nprocs = 16
perm = args.perm
dataset = args.dataset
gentype = args.gentype
nprocs = args.nprocs

rdmacallfile = join(abspath,f"rdmacalls_nprocs{nprocs}_dataset{dataset}_gentype{gentype}_{perm}.bin")
rdmacalls = np.fromfile(rdmacallfile,dtype=np.int32)
rdmacalls = rdmacalls.reshape((-1,nprocs))
rdmamemfile = join(abspath,f"rdmamems_nprocs{nprocs}_dataset{dataset}_gentype{gentype}_{perm}.bin")
rdmamems = np.fromfile(rdmamemfile,dtype=np.float64)
rdmamems = rdmamems.reshape((-1,nprocs))
rdmatimefile = join(abspath,f"rdmatimes_nprocs{nprocs}_dataset{dataset}_gentype{gentype}_{perm}.bin")
rdmatimes = np.fromfile(rdmatimefile,dtype=np.float64)
rdmatimes = rdmatimes.reshape((-1,nprocs))
with open(f"rdmacalls_nprocs{nprocs}_dataset{dataset}_gentype{gentype}_{perm}.txt", 'w') as f:
    f.write(f"RDMA Call Number Analysis, Dataset:{dataset}, gentype{gentype}, nprocs{nprocs} Source is Row, Target is Column\n")
    np.savetxt(f,rdmacalls,fmt="%6d")
    f.write(f"\n\nMemory Requirement Analysis, Dataset:{dataset}, gentype{gentype}, nprocs{nprocs} Source is Row, Target is Column\n")
    np.savetxt(f,rdmamems,fmt="%7.2f")
    f.write(f"\n\nRDMA Time Analysis (unit: seconds), Dataset:{dataset}, gentype{gentype}, nprocs{nprocs} Source is Row, Target is Column\n")
    np.savetxt(f,rdmatimes,fmt="%7.3f")

