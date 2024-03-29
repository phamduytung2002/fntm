@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Define the arrays
set "array1=100 250 "
set "array2=1 10 100 250 "

REM Loop through the first array
for %%e in (%array1%) do (
    REM Loop through the second array
    for %%x in (%array2%) do (
        echo %%e %%x
        python main.py --dataset 20NG --model XTM --num_topics 50 --epochs 500 ^
            --device cuda --lr 0.002 --lr_scheduler StepLR --dropout 0 ^
            --batch_size 200 --lr_step_size 125 --use_pretrainWE --seed 0 ^
            --weight_ECR %%e --weight_XGR %%x
    )
)

ENDLOCAL
