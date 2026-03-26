
python annotator.py --input_path ../../OJBench/results/evaluation_datas/math/qwen3_8b/part_0.jsonl --output_path annotated/math/student/train_set.jsonl --model_name Qwen/Qwen3-32B

python annotator.py --input_path ../../OJBench/results/evaluation_datas/math/gpt_oss/part_0.jsonl  --output_path annotated/math/teacher/train_set.jsonl --model_name Qwen/Qwen3-32B
