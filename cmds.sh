nohup uv run python main.py --model_name Qwen/Qwen2.5-0.5B --debug_mode --sample_size 10000 --epochs 1 --batch_size 1 --learning_rate 1e-3 --force_reload > training.log 2>&1 &

scp -p 11152 -i 6rmuhtx0rsb1si -r root@99.69.17.69:root/mlx-week-6/trained_model ./