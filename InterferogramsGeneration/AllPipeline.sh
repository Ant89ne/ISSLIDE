#!/usr/bin/env bash


configFile="/Users/antoinebralet/Desktop/Research/Code/ISSLIDE/InterferogramsGeneration/config/configAll.json"

python3 splitSwaths.py --config $configFile

echo ""
echo "End of Splitting"
echo ""

python3 GenerateInterf.py --config $configFile

echo ""
echo "End of Interferogram Generation"
echo ""

python3 OrthorectInterf.py --config $configFile

echo ""
echo "End of Orthorectification"
echo ""
