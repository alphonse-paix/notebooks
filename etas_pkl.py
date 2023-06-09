import os
import pandas as pd
import pickle
import shutil
import subprocess
import sys
from pathlib import Path


def etas(args=""):
    # data directory
    cwd = "data"
    try:
        os.mkdir(cwd)
    except:
        pass

    # program run
    path = Path(r"C:\Users\Alphonse\Documents\ETAS\x64\Release\ETAS.exe")
    p = subprocess.Popen([path] + args.split(),
                            stdout=subprocess.PIPE,
                            cwd=cwd,
                            universal_newlines=False)

    # output
    stdout = open(os.dup(p.stdout.fileno()), newline="")
    for line in stdout:
        print(line, end="") 


def gen_dataset(args="", save=False, filename="dataset.pkl"):
    dirname = "data_temp_seqs"
    etas("--generate_seqs --dirname " + dirname + " " + args)

    os.chdir("data/" + dirname)
    seqs = []
    for file in os.listdir():
        data = pd.read_csv(file, index_col=0)
        seqs.append(data)
        
    if save:
        with open("../" + filename, "wb") as f:
            pickle.dump(seqs, f)

    os.chdir("..")
    shutil.rmtree(dirname)
    os.chdir("..")

    return seqs