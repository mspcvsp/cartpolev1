import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
import time
import wandb
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name",
                        type=str,
                        default=Path(os.path.abspath(__file__)).parts[-2],
                        help="the name of this experiment")

    parser.add_argument("--gym-id",
                        type=str,
                        default="CartPole-v1",
                        help="the id of the gym environment")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=2.5e-4,
                        help="the learning rate of the optimizer")
    
    parser.add_argument("--seed",
                        type=int,
                        default=302756697,
                        help="seed of the experiment")

    parser.add_argument("--total-timesteps",
                        type=int,
                        default=25000,
                        help="total timesteps of the experiments")

    parser.add_argument("--torch-deterministic",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="if toggled, " +\
                        "`torch.backends.cudnn.deterministic=False`")

    parser.add_argument("--cuda",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="if toggled, cuda will be enabled by default")

    parser.add_argument("--track",
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs="?",
                        const=False,
                        help="if toggled, this experiment will be" +\
                        " tracked with Weights and Biases")
 
    parser.add_argument("--wandb-project-name",
                        type=str,
                        default="ppo-implementation-details",
                        help="the wandb's project name")

    parser.add_argument("--wandb-entity",
                        type=str,
                        default="mspcvsp94",
                        help="the entity (team) of wandb's project")

    parser.add_argument("--capture-video",
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs="?",
                        const=True,
                        help="whether to capture videos of the agent" +\
                        "performances (check out `videos` folder)")

    """
    Algorithm specific arguments
    """
    parser.add_argument("--num-envs",
                        type=int,
                        default=4,
                        help="the number of parallel game environments")

    parser.add_argument("--num-steps",
                        type=int,
                        default=500,
                        help="the number of steps to run in each " +\
                        "environment per policy rollout")

    parser.add_argument("--anneal-lr",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="Toggle learning rate annealing for " +\
                        "policy and value networks")

    parser.add_argument("--gae",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="Use GAE for advantage computation")

    parser.add_argument("--gamma",
                        type=float, default=0.99,
                        help="the discount factor gamma")
    
    parser.add_argument("--gae-lambda",
                        type=float,
                        default=0.95,
                        help="the lambda for the general " +\
                        "advantage estimation")

    parser.add_argument("--num-minibatches",
                        type=int,
                        default=4,
                        help="the number of mini-batches")

    parser.add_argument("--update-epochs",
                        type=int,
                        default=4,
                        help="the K epochs to update the policy")

    parser.add_argument("--norm-adv",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="Toggles advantages normalization")

    parser.add_argument("--clip-coef",
                        type=float,
                        default=0.2,
                        help="the surrogate clipping coefficient")

    parser.add_argument("--clip-vloss",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="Toggles whether or not to use a clipped "+ \
                        "loss for the value function, as per the paper.")

    parser.add_argument("--ent-coef",
                        type=float,
                        default=0.01,
                        help="coefficient of the entropy")

    parser.add_argument("--vf-coef",
                        type=float,
                        default=0.5,
                        help="coefficient of the value function")

    parser.add_argument("--max-grad-norm",
                        type=float,
                        default=0.5,
                        help="the maximum norm for the gradient clipping")

    parser.add_argument("--target-kl",
                        type=float,
                        default=None,
                        help="the target KL divergence threshold")

    return parser.parse_args()

def initialize(**kwargs):

    args = parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    args.device =\
        torch.device("cuda" if torch.cuda.is_available()
                     and args.cuda else "cpu")

    seconds_since_epoch =\
        kwargs.get("seconds_since_epoch",
                   int(time.time()))

    args.run_name =\
        f"{args.gym_id}__{args.exp_name}__{args.seed}__{seconds_since_epoch}"

    if args.track:

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{args.run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %
        ("\n".join([f"|{key}|{value}|"
                    for key, value in vars(args).items()])),
    )

    torch.backends.cudnn.deterministic = args.torch_deterministic

    return args, writer


def initialize_rngs(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seed_seq = SeedSequence(args.seed)

    return Generator(MT19937(seed_seq))
