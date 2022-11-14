#!/usr/bin/bash



NAME=$1 #Molecule Name, should be a NAME.hdf5 file in /data
ITERS=10000 #Number of optimization iterations
SPEC=NADE #Specify saving directory
ELECTRONS=$2 # Number of electrons
SPIN=$3 # Multiplicity
ORBITALS=$4 # Spatial Orbitals
SAMPLES=10000 # Unique Samples
LR=0.001 # learning rate
SEED=$5 # Random Seed
SUBSIZE=1 #Number of spatial orbitals per subnetwork

echo "Running calculation in calcs/${NAME}_$SPEC"
echo "Molecule $NAME 
Electrons $ELECTRONS 
Multiplicity $SPIN 
Spatial Orbitals $ORBITALS 
Seed $SEED"
echo "Further options can be viewed via python3 scripts/nade.py"

mkdir "calcs/${NAME}_$SPEC"
cd "calcs/${NAME}_$SPEC"

nohup python3 -u -m scripts.nade -n $NAME -i $ITERS -s $SAMPLES -o $ORBITALS -m $SPIN -e $ELECTRONS -lr $LR -se $SEED -sub $SUBSIZE > "${NAME}_${SEED}.out" &

echo "Output is written to ${NAME}_${SEED}.out"
echo "Check calculation via scripts/energy.py LOGFILENAME, consider adding /scripts to your path"
echo "Plot your results via scripts/plot.py -f LOGFILENAME -r REFERENCEENERGY"
