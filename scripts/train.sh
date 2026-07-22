DATA=$1
Batch=$2
GPU=$3
EXP=$4

dirname="model_para_ckpt/${DATA}/${EXP}"
mkdir -p -- "$dirname"
python3 -m tools.train --config config_files/${DATA}.yaml --batch_size ${Batch} \
					 		--gpus ${GPU} --exp ${EXP} --enc spt --num_tokens 10 --patch_size 16 --prompt plural --con rank \
							    | tee -a ${dirname}/log.txt