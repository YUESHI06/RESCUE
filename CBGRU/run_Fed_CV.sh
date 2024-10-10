noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
vuls=("reentrancy" "timestamp")


for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..5}
    do
        python Fed_CV.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate  --device cuda:4 --batch 8 --device cuda:3
    done
    done
done