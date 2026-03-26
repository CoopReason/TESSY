
# python train_predictor.py --input_path annotated/math/student/train_set.jsonl --output_path  --model_name Qwen/Qwen3-32B

# python train_predictor.py --input_path annotated/math/teacher/train_set.jsonl  --output_path  --model_name Qwen/Qwen3-32B

CUDA_VISIBLE_DEVICES=0 python train_predictor.py --data_path annotated/math/student/train_set.jsonl --output_dir checkpoints/math/student

CUDA_VISIBLE_DEVICES=1 python train_predictor.py --data_path annotated/math/teacher/train_set.jsonl --output_dir checkpoints/math/teacher
