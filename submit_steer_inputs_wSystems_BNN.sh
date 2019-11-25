#!/bin/bash
#SBATCH --partition=cms-uhh,cms,allgpu
#SBATCH --time=1-00:00:00                           # Maximum time requested
#SBATCH --constraint=GPU
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --job-name  steer
#SBATCH --output    steer-%N-%j.out            # File to which STDOUT will be written
#SBATCH --error     steer-%N-%j.err            # File to which STDERR will be written
#SBATCH --mail-type ALL                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user ksenia.de.leo@desy.de          # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.
#SBATCH --requeue

source .setenv
echo $PYTHONPATH
cd /beegfs/desy/user/deleokse/ZprimeClassifier
./steer_inputs_wSystems_BNN.py
