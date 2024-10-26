#!/bin/bash

help(){
   # Display Help
   echo "Scripts for fast extracting nearby signals for candidate 6mA sites."
   echo "Args can be added [-b|r|f|g|o|p|w|s|h]"
   echo "options:"
   echo "b     Full bam file for processing."
   echo "r     Region of 6mAs to be extracted, in bed format (0-based, strand specific), ignore if -f provided."
   echo "f     Output file from step1, will analyzed on 6mA sites, ignore if -r provided."
   echo "g     Reference genome."
   echo "o     Output_dir. (DEFAULT: current working dir)."
   echo "p     Pre of the output file (DEFAULT:output)."
   echo "w     Number of workers to use (DEFAULT:10)."
   echo "s     Print the script dir.(DEAFULT:cwd)"
   echo "h     Print help documentation."
}

################
# Main program #
################
# Set variables by default.
num_workers=10
scripts_dir=$PWD

#####################################################
# Process the input options. Add options as needed. #
#####################################################
# Get the options
while getopts ":b:r:f:g:o:p:w:s:" option; do
case $option in
      b) bam=$OPTARG ;;
      r) bed=$OPTARG ;;
      f) file=$OPTARG ;;
      g) genome=$OPTARG ;;
      o) output_dir=$OPTARG ;;
      p) pre=$OPTARG ;;
      w) num_workers=$OPTARG ;;
      s) scripts_dir=$OPTARG ;;
      *) echo "Invalid parameter!" ;;
   esac
done

## Check mandatory argument
if [ $# == 0 ] || [ $1 == "-h" ] || [ $1 == "--help" ];then
   help
   exit 1
fi

[[ -f $bam  ]] || { help;echo -e "\n[ERR] Bam file not exist!";exit; }
[[ -f $file && -f $bed ]] && { help;echo -e "\n[ERR] Conflict! only profile a region bed or output from step1";exit; }

#####################
# Run main process. #
#####################
if [ $output_dir ];then
   mkdir -p $output_dir
else
   output_dir=$(dirname $PWD)
fi

cd $output_dir
echo ${output_dir}

if [ $pre ];then
   :
else
   pre="output"
fi

if [ $file ];then
   awk '$11=="6mA"{print $2}' $file | awk -F '_' '{gsub(1,"+",$2);gsub(0,"-",$2);{print $1 "\t" $3 "\t" $3+1 "\tname\tnum\t" $2}}' > ${pre}.bed
   bed=${pre}.bed
elif [ $bed ];then
   :
else
   { help;echo -e "\n[ERR] No region file provided!";exit; }
fi

awk '{gsub("+",1,$6);gsub("-",0,$6);print $1 "," $2 "," $6}' $bed > ${pre}.csv

# samtools: extract
samtools view -@ $num_workers -h -b -L $bed \
$bam > ${pre}.bam && samtools index ${pre}.bam && pbindex ${pre}.bam && \

# python file: extract
python $scripts_dir/main.py ${pre}.bam \
--sitesFile ${pre}.csv \
--reference $genome \
--sector 10 \
--outputFile ${pre} \
--numWorkers $num_workers \
--minCoverage 5 && echo "Extraction finshed."
