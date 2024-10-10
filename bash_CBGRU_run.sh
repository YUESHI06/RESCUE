# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.0)
vuls=("reentrancy" "timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..10}
    do
        python Fed_CBGRU.py --vul $vul --noise_type pure --noise_rate $noise_rate --cbgru_net2 bigru --device cuda:5
    done
    done
done