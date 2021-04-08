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

source .setenv_V4
export PATH="/beegfs/desy/user/deleokse/anaconda2/bin:$PATH" 
source activate py27_Zprime 
echo $PYTHONPATH
cd /beegfs/desy/user/deleokse/ZprimeClassifier_V4_HOTVR
./steer_inputs_wSystems_DNN_V4.py
