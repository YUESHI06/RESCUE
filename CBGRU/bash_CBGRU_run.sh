# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..10}
    do
        python Fed_Avg.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate --cbgru_net2 bilstm --device cuda:7
    done
    done
done