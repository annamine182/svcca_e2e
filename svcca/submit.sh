#!/bin/bash
# Md Asif Jalal,The University of Sheffield,2021 

wdir=/share/mini1/sw/spl/espresso/new_svcca/svcca
jid="-"
killscript=$wdir/killall.sh
python=/share/mini1/sw/std/python/anaconda3-2020.11/2020.11/envs/asr_e2e/bin/python
submitjob=/share/mini1/sw/mini/jet/latest/tools/submitjob

#####log handling###

rm $wdir/log/*

###GPU specification####

number_of_submissions=1
gpu='GeForceGTX1080Ti|GeForceGTXTITANX'
number_of_gpus=1
number_of_threads=2
memory_size=60000

#########################
### Script Parameters ###

start_stage=2
end_stage=8

#########################

## process tracker and killscript##

echo "#!/bin/bash" > ${killscript}

##########
### run script path ###

#S=${wdir}/test.sh
#S=${wdir}/print.sh
#S=${wdir}/script.sh
S=${wdir}/run.sh
#S=${wdir}/transformer_run.sh

echo " Running Script $S"
#########
for ((i=0;i<$number_of_submissions;i++))
do
  L=${wdir}/log/log_${i}.log
  if [ "$jid" == "-" ]; then
    jid=`$submitjob -g$number_of_gpus -M$number_of_threads -o -l gputype=$gpu -eo $L $S | tail -1`
  else
    jid=`$submitjob -g$number_of_gpus -M$number_of_threads -o -l gputype=$gpu -eo -w $jid $L $S | tail -1`
  fi
  echo "$S $L $jid"
  echo "qdel $jid" >> ${killscript}
done


