time python main.py \
demo/WT.chr11_70000000_70100000.mm10.bam \
--control demo/WGA.chr11_70000000_70100000.mm10.bam \
--reference demo//mm10.fa \
--outputFile demo/WT.chr11_70000000_70100000.6mA \
-w chr11:70000000-70100000 \
--referenceStride 10000 \
--numWorkers 10 \
--outputType full 1> demo/run.log 2>demo/err.log &
# memory consuming: ~400-500M/CPU
# real    4m6.229s
# user    29m54.064s
# sys     2m24.715s
