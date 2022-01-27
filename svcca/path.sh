MAIN_ROOT=$PWD/..
KALDI_ROOT=$MAIN_ROOT/espresso/tools/kaldi

# BEGIN from kaldi path.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8
# END

export PATH=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/asr_e2e/bin:$PATH
export PATH=$MAIN_ROOT:$MAIN_ROOT/espresso:$MAIN_ROOT/espresso/tools:$PATH
export PYTHONPATH=$MAIN_ROOT:$MAIN_ROOT/espresso:$MAIN_ROOT/espresso/tools:$PYTHONPATH
export PYTHONUNBUFFERED=1


#export
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/include/:/share/mini1/sw/std/cuda/cuda10.1/cuda/:/share/mini1/sw/std/cuda/cuda10.1/x86_64/lib64/stubs

#for nist scoring
export SCTK=$/share/mini1/sw/spl/sctk/v2.4.8/sctk-2.4.8/

# for RTX3090
#export PATH=/share/mini1/sw/std/python/anaconda3-2020.11/2020.11/envs/e2e/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/share/mini1/sw/std/cuda/cuda11.2/x86_64/lib64/:/share/mini1/sw/std/cuda/cuda11.2/x86_64/include/:/share/mini1/sw/std/cuda/cuda11.2/cuda/:/share/mini1/sw/std/cuda/cuda11.2/x86_64/lib64/stubs

 
