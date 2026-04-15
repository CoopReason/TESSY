

CUDA_VISIBLE_DEVICES=0 python train_predictor.py --data_path annotated/code/student/train_set.jsonl --output_dir checkpoints/code/student

CUDA_VISIBLE_DEVICES=0 python train_predictor.py --data_path annotated/code/teacher/train_set.jsonl --output_dir checkpoints/code/teacher
