python main.py --dataset 20NG --model XTMv3 --num_topics 50 --epochs 200 ^
--device cuda --lr 0.001 --lr_scheduler StepLR --dropout 0 ^
--batch_size 200 --lr_step_size 125 --use_pretrainWE --seed 3 ^
--weight_ECR 250 --weight_XGR 250 --alpha_ECR 20 --alpha_XGR 5
