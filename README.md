# Identify 6mA sites in mammals using 6mA-SniperPro

## 0. Installation
After our paper publish, you can directly install 6mA-SniperPro via pip/conda, please wait for a while, it's coming soon.

## 1. Identification
First, you should prepare SMRT-seq data from PacBio Sequel II platform, then use the main.py in **Identification**:
>usage: main.py 
>>alignment_set --control CONTROL  --reference REFERENCE --outputFile OUTFILE [options]
>
>> options:
>>>[-h] [--version] [--log-file LOG_FILE] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL} | --debug | -->quiet | -v] 
>>> [--outputType OUTTYPE] [--numWorkers NUMWORKERS] [--maxAlignments MAXALIGNMENTS]  
>>> [--sigQcutoff SIGQCUTOFF] [--minCoverage MINCOVERAGE] [--minCoverage_site MINCOVERAGE_SITE]
>>>  [-w >REFERENCEWINDOWSASSTRING]  [-W REFERENCEWINDOWSASSTRING] [--seed RANDOMSEED]
>>>   [--referenceStride REFERENCESTRIDE] [--sector >SECTOR] [--maxQueueSize MAXQUEUESIZE] 
>>>  [--modelDict MODELDICT] [--modelPath MODELPATH] [--COVERAGE_model COVERAGE_MODEL]
>>>  [--FREQ_model FREQ_MODEL]  [--THRESHOLD_model THRESHOLD] [--dip_P DIPP] [--ks2_P KS2P]  

For more concrete instructions, you can tape 'python main.py -h' for complete guidance.
Note: before using this script, three things should be ready:
1) Reconstruct the virtual environment in anaconda using requirements.txt
2) Install pbcore (https://github.com/PacificBiosciences/pbcore) & download pbcommand (https://github.com/PacificBiosciences/pbcommand) from PacBio beforehand, save them to where you stored this folder.
3) Change '--modelPath' above to where you save the model_ccs_attn.py, and '--modelDict' to where you save the state_dict.pt.
   P.S.: Those files are available in **model**.
