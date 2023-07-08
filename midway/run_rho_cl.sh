sitenumarray=(10 25)
xiarray=(0.1 1.0 10.0 100.0)
pfarray=(20.76)
paarray=(44.75)
thetaarray=(1.0)
gammaarray=(1.0)
timearray=(200)
weightarray=(0.25 0.5 0.75 1.0)
mix_in_array=(2 3 4 5 6 7 8 9 10)
mass_matrix_array=(1 5 10)
step_size_array=(2 3 4 5 6 7 8 9 10)

hmc_python_name="sampler.py"

for sitenum in "${sitenumarray[@]}"; do
    for xi in "${xiarray[@]}"; do
        for pf in "${pfarray[@]}"; do
            for pa in "${paarray[@]}"; do
                for time in "${timearray[@]}"; do
                    for weight in "${weightarray[@]}"; do
                        for theta in "${thetaarray[@]}"; do
                            for gamma in "${gammaarray[@]}"; do
                                    count=0
                                                
                                    action_name="test5"

                                    dataname="${action_name}"

                                    mkdir -p ./job-outs/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/

                                    if [ -f ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.sh ]; then
                                        rm ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.sh
                                    fi

                                    mkdir -p ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/

                                    touch ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.sh

                                    tee -a ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.sh <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=${theta}_${gamma}_sn_${sitenum}_xi_${xi}_w_${weight}_${action_name}
#SBATCH --output=./job-outs/$job_name/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.out
#SBATCH --error=./job-outs/$job_name/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.err
#SBATCH --time=1-11:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=12G

module load python/anaconda-2022.05
  

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)

source /project/lhansen/hmc/Gams/env/bin/activate
python3 -u /project/lhansen/hmc/$hmc_python_name  --pf ${pf} --pa ${pa} --time ${time} --theta ${theta} --gamma ${gamma} --sitenum ${sitenum} --xi ${xi} --weight ${weight} --dataname ${dataname}
echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                                count=$(($count + 1))
                                sbatch ./bash/${action_name}/pf_${pf}_pa_${pa}_time_${time}/theta_${theta}_gamma_${gamma}/sitenum_${sitenum}_xi_${xi}/weight_${weight}/run.sh
                            done
                        done
                    done
                done
            done
        done
    done
done