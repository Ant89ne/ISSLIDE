#!/usr/bin/env bash


configFile="/Users/antoinebralet/Desktop/Research/Code/ISSLIDE/InterferogramsGeneration/config/configAll.json"

python3 splitSwaths.py --config $configFile

echo "End of Splitting"

python3 GenerateInterf.py --config $configFile

echo "End of Interferogram Generation"

python3 OrthorectInterf.py --config $configFile

echo "End of Orthorectification"