# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

#export train_cmd="run.pl --mem 4G"
#export cuda_cmd="run.pl --mem 4G --gpu 1"
#export decode_cmd="run.pl --mem 4G"

# JHU setup (copy queue-freegpu.pl from ESPnet into utils/)
#export train_cmd="queue.pl --mem 4G"
#export cuda_cmd="queue-freegpu.pl --mem 8G --gpu 1 --config conf/gpu.conf"
#export decode_cmd="queue.pl --mem 4G"

# Sheffield setup
export train_cmd="run.pl --mem 5G"
export decode_cmd="run.pl --mem 5G"
export cuda_cmd="/share/mini1/sw/mini/jet/latest/tools/submitjob $cmdgpu"

cmdstorage="-p MINI -q NORMAL -m 4000  -o -l hostname=node22|node23|node24|node25|node26 -eo"
cmdstoragelarge="-p MINI -q NORMAL -m 5000  -o -l hostname=node22|node23|node24|node25|node26 -eo"
cmdstorageextralarge="-p MINI -q NORMAL -m 10000  -o -l hostname=node22|node23|node24|node25|node26 -eo"
cmdnormal="-p MINI -q NORMAL -m 4000  -o -l hostname=node22|node23|node24|node25|node26 -eo"
cmdnormallarge="-p MINI -q NORMAL -m 5000  -o -l hostname=node22|node23|node24|node25|node26 -eo"
cmdextralarge="-p MINI -q NORMAL -m 10000  -o -l hostname=node22|node23|node24|node25|node26 -eo"
cmdhuge="-p MINI -q NORMAL -m 30000  -o -l hostname=node24|node25|node26 -eo"
cmdgpu="-p MINI -q GPU -o -l hostname=node24|node25 -eo"

