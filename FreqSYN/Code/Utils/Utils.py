import json


def denorm(x, norm):
    norm_min = norm[0]
    norm_max = norm[1]
    return 0.5 * (x + 1) * (norm_max - norm_min) + norm_min


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
def add_args(parser):
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--train_name", type=str, help="The name of the training")
    parser.add_argument("--action_dim", type=str, default="6", help="action dimention")
    parser.add_argument("--train_cfg", type=str, help="train config")

def load_cfg_args(args, cfg):
    args.max_train_steps = cfg.max_train_steps
    args.evaluate_freq = cfg.evaluate_freq
    args.save_freq = cfg.save_freq
    args.policy_dist = cfg.policy_dist
    args.batch_size = cfg.batch_size
    args.mini_batch_size = cfg.mini_batch_size
    args.lr_a = cfg.lr_a
    args.lr_c = cfg.lr_c
    args.gamma = cfg.gamma
    args.lamda = cfg.lamda
    args.epsilon = cfg.epsilon
    args.K_epochs = cfg.K_epochs
    args.evaluate_times = cfg.evaluate_times
    args.device = cfg.device

    