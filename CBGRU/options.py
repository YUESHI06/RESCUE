import argparse

def parse_args():
    parser = argparse.ArgumentParser("Local model Experiments")
    
    parser.add_argument(
        '--vul',
        default='reentrancy',
        type=str,
        help='Type of vulnerability'
    )

    parser.add_argument(
        '--epoch',
        default=50,
        type=int,
        help='epochs for global traning'
    )

    parser.add_argument(
        '--local_epoch',
        default=1,
        type=int,
        help='epochs for local training'
    )

    parser.add_argument(
        '--inner_lr',
        default=0.0005,
        type=float,
        help="learning rate for inner model"
    )

    parser.add_argument(
        '--outer_lr',
        default=0.0003,
        type=float,
        help='learning rate for outer model'
    )

    parser.add_argument(
        '--batch',
        default=8,
        type=int,
        help='batch size for data loader'
    )

    parser.add_argument(
        '--input_channels',
        default=428,
        type=int,
        choices=[138, 428],
        help='input channels for LCN model'
    )

    parser.add_argument(
        '--client_num',
        default=8,
        type=int,
        help='num of clients in federated training'
    )

    parser.add_argument(
        '--noise',
        action='store_true'
    )

    parser.add_argument(
        '--noise_type',
        choices=['pure', 'noise', 'fn_noise', 'non_noise', 'diff_noise'],
        default='noise',
        help='Is dataset contains noise. If it does, what kind of noise is'
    )

    parser.add_argument(
        '--alpha',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--beta',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--noise_rate',
        default=0.05,
        type=float
    )

    parser.add_argument(
        '--device',
        default='cuda:0',
        type=str,
        help="training device"
    )

    parser.add_argument(
        '--cbgru_local_epoch',
        default=1,
        type=int
    )

    parser.add_argument(
        '--cbgru_local_lr',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '-d',
        '--dropout', 
        type=float, 
        default=0.5, 
        help='dropout rate')
    
    parser.add_argument(
        '--cbgru_net1',
        type=str,
        default='cnn',
        choices=['cnn', 'bilstm', 'bigru']
    )

    parser.add_argument(
        '--cbgru_net2',
        type=str,
        default='bigru',
        choices=['cnn', 'bigru', 'bilstm']
    )

    parser.add_argument(
        '--sample_rate',
        type=float,
        default=0.4,
        help="client sample rate"
    )

    parser.add_argument(
        '--seed',
        type=float,
        default=1.
    )

    parser.add_argument(
        '--relabel_ratio', 
        type=float, 
        default=0.5, 
        help="proportion of relabeled samples among selected noisy samples"
    )

    parser.add_argument(
        '--fine_tuning', 
        action='store_false', 
        help='whether to include fine-tuning stage'
    )
    
    parser.add_argument(
        '--correction', 
        action='store_false', 
        help='whether to correct noisy labels'
    )

    parser.add_argument(
        '--reg_weight', 
        help='weight of regularization term', 
        type=float, 
        required=False
    )

    parser.add_argument(
        '--frac2',
        type=float,
        default=0.1,
        help="fration of selected clients in fine-tuning and usual training stage"
    )

    parser.add_argument(
        '--rounds1', 
        type=int, 
        default=200, 
        help="rounds of training in fine_tuning stage"
    )

    parser.add_argument(
        '--rounds2', 
        type=int, 
        default=200, 
        help="rounds of training in usual training stage"
    )
    
    parser.add_argument(
        '--corr_seed',
        type=int,
        default=13
    )

    parser.add_argument(
        '--iteration1',
        type=int,
        default=50,
        help="enumerate iteration in preprocessing stage"
    )

    parser.add_argument(
        '--confidence_thres', 
        type=float, 
        default=0.5, 
        help="threshold of model's confidence on each sample"
    )

    parser.add_argument(
        '--clean_set_thres', 
        type=float, 
        default=0.1, 
        help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage"
    )

    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help="num of classes for classification"
    )

    parser.add_argument(
        '--first_epochs', 
        type=int, 
        default=50,
        help="number of rounds before correction"
    )

    parser.add_argument(
        '--last_epochs', 
        type=int, 
        default=50,
        help="number of rounds after correction"
    )

    parser.add_argument(
        '--warm_up_epoch', 
        type=int, 
        default=10,
        help="warmp up epochs for PLE"
    )

    parser.add_argument(
        '--valid_frac',
        type=float,
        default=1.0,
        help="fraction of how much test data uesd to test"
    )

    args = parser.parse_args()
    return args