DATA=$1
GPU=$2
EXP=$3

dirname="model_ckpt/${DATA}/${EXP}/inference"
mkdir -p -- "$dirname"
python3 -m tools.test --config config_files/${DATA}.yaml \
					 		--gpus ${GPU} --exp ${EXP} --enc spt --num_tokens 10 --patch_size 16 --prompt plural --ckpt_used best \
							    | tee ${dirname}/log.txt
