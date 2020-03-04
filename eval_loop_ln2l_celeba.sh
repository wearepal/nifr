######################################################################
# Options for the batch system
# These options are not executed by the script, but are instead read by the
# batch system before submitting the job. Each option is preceeded by '#$' to
# signify that it is for grid engine.
#
# All of these options are the same as flags you can pass to qsub on the
# command line and can be **overriden** on the command line. see man qsub for
# all the details
######################################################################
# -- The shell used to interpret this script
#$ -S /bin/bash
# -- Execute this job from the current working directory.
#$ -cwd
# -- use the short.q
##$ -q short.q
# -- Job output to stderr will be merged into standard out. Remove this line if
# -- you want to have separate stderr and stdout log files
#$ -j y
#$ -o ../output/
# -- Send email when the job exits, is aborted or suspended
##$ -m eas
##$ -M PUT_YOUR_USERNAME_HERE_TO_ENABLE_EMAIL_NOTIFICATION@sussex.ac.uk
##$ -pe mpi 6

######################################################################
# Job Script
# Here we are writing in bash (as we set bash as our shell above). In here you
# should set up the environment for your program, copy around any data that
# needs to be copied, and then execute the program
######################################################################
echo "nvidia-smi"
nvidia-smi
echo "-------------------"
echo "Starting job script"
echo "-------------------"
username=`whoami`
python_exe=/mnt/data/tk324/miniconda3/bin/python
echo "User: $username"
export WANDB_API_KEY="73446d9c90d14d0603dbefef971c2cd923ad7f4e"

for mf in "0.0" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "0.95" "1.0"
do
	$python_exe -u start_ln2l.py \
	--dataset celeba --task-mixing-factor $mf \
	--lr 1e-3 --batch-size 128 --weight-decay 0 --epochs 25 \
        --entropy-weight 0.01 \
        --results-csv ln2l_celeba "$@"
done

echo "Finished job script"

