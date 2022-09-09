"""Main training script for RL^2."""
import argparse
import collections
import os
import shutil

import numpy as np
import torch
import tqdm

import config as cfg
import dqn
import dream
import relabel
import utils
import wrappers


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            "-c", "--configs", action="append", default=["configs/rl2.json"])
    arg_parser.add_argument(
            "-b", "--config_bindings", action="append", default=[],
            help="bindings to overwrite in the configs.")
    arg_parser.add_argument(
            "-x", "--base_dir", default="experiments",
            help="directory to log experiments")
    arg_parser.add_argument(
            "-p", "--checkpoint", default=None,
            help="path to checkpoint directory to load from or None")
    arg_parser.add_argument(
            "-f", "--force_overwrite", action="store_true",
            help="Overwrites experiment under this exp name, if it exists.")
    arg_parser.add_argument(
            "-s", "--seed", default=0, help="random seed to use.", type=int)
    arg_parser.add_argument(
            "-t", "--steps", default=int(5e5),
            help="maximum number of steps to train for.", type=int)
    arg_parser.add_argument("exp_name", help="name of the experiment to run")
    args = arg_parser.parse_args()
    config = cfg.Config.from_files_and_bindings(
            args.configs, args.config_bindings)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_dir = os.path.join(os.path.expanduser(args.base_dir), args.exp_name)
    if os.path.exists(exp_dir) and not args.force_overwrite:
        raise ValueError(f"Experiment already exists at: {exp_dir}")
    shutil.rmtree(exp_dir, ignore_errors=True)  # remove directory if exists
    os.makedirs(exp_dir)

    # pylint: disable=unspecified-encoding
    with open(os.path.join(exp_dir, "config.json"), "w+") as f:
        # pylint: enable=unspecified-encoding
        config.to_file(f)
    print(config)

    env_class = dream.get_env_class(config.get("environment"))

    # pylint: disable=unspecified-encoding
    with open(os.path.join(exp_dir, "metadata.txt"), "w+") as f:
        # pylint: enable=unspecified-encoding
        f.write(f"Split: {env_class.env_ids()}\n")

    # Use GPU if possible
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda:0")

    print(f"Device: {device}")
    tb_writer = utils.EpisodeAndStepWriter(os.path.join(exp_dir, "tensorboard"))

    text_dir = os.path.join(exp_dir, "text")
    os.makedirs(text_dir)

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir)

    create_env = env_class.create_env
    exploration_env = create_env(0)
    instruction_env = env_class.instruction_wrapper()(exploration_env, [])
    multi_episode_env = wrappers.MultiEpisodeWrapper(instruction_env)
    agent = dqn.DQNAgent.from_config(
            config.get("agent"), multi_episode_env)

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load_state_dict(
                torch.load(os.path.join(args.checkpoint, "agent.pt")))

    rewards = collections.deque(maxlen=200)
    episode_lengths = collections.deque(maxlen=200)
    total_steps = 0
    for episode_num in tqdm.tqdm(range(1000000)):
        exploration_env = create_env(episode_num)
        instruction_env = env_class.instruction_wrapper()(
                exploration_env, [], seed=episode_num + 1,
                first_episode_no_optimization=True)
        multi_episode_env = wrappers.MultiEpisodeWrapper(instruction_env, 2)

        # Switch between IDs and not IDs for methods that use IDs
        # Otherwise, no-op
        if episode_num % 2 == 0:
            # pylint: disable=protected-access
            if hasattr(agent._dqn._q._state_embedder, "use_ids"):
                agent._dqn._q._state_embedder.use_ids(True)
            # pylint: enable=protected-access

        episode, _ = dream.run_episode(multi_episode_env, agent)

        for index, exp in enumerate(episode):
            agent.update(relabel.TrajectoryExperience(exp, episode, index))

        # pylint: disable=protected-access
        if hasattr(agent._dqn._q._state_embedder, "use_ids"):
            agent._dqn._q._state_embedder.use_ids(False)
        # pylint: enable=protected-access

        total_steps += len(episode)
        episode_lengths.append(len(episode))
        rewards.append(sum(exp.reward for exp in episode))

        if episode_num % 100 == 0:
            for k, v in agent.stats.items():
                if v is not None:
                    tb_writer.add_scalar(
                            f"agent_{k}", v, episode_num, total_steps)

            tb_writer.add_scalar(
                    "steps/total", total_steps, episode_num, total_steps)
            tb_writer.add_scalar(
                    "reward/train", np.mean(rewards), episode_num, total_steps)
            tb_writer.add_scalar(
                    "steps/steps_per_episode", np.mean(episode_lengths),
                    episode_num, total_steps)

        if episode_num % 2000 == 0:
            visualize_dir = os.path.join(exp_dir, "visualize", str(episode_num))
            os.makedirs(visualize_dir, exist_ok=True)

            test_rewards = []
            test_episode_lengths = []
            for test_index in tqdm.tqdm(range(100)):
                exploration_env = create_env(test_index, test=True)
                instruction_env = env_class.instruction_wrapper()(
                        exploration_env, [], seed=test_index + 1, test=True,
                        first_episode_no_optimization=True)
                multi_episode_env = wrappers.MultiEpisodeWrapper(
                        instruction_env, 2)
                episode, render = dream.run_episode(
                        multi_episode_env, agent, test=True)
                test_episode_lengths.append(len(episode))

                test_rewards.append(sum(exp.reward for exp in episode))

                if test_index < 10:
                    frames = [frame.image() for frame in render]
                    save_path = os.path.join(
                            visualize_dir, f"{test_index}.gif")
                    frames[0].save(save_path, save_all=True,
                                   append_images=frames[1:], duration=750,
                                   loop=0)

            tb_writer.add_scalar(
                    "reward/test", np.mean(test_rewards), episode_num,
                    total_steps)
            tb_writer.add_scalar(
                    "steps/test_steps_per_episode",
                    np.mean(test_episode_lengths), episode_num, total_steps)

            if total_steps > args.steps:
                return

        if episode_num != 0 and episode_num % 20000 == 0:
            print("Saving checkpoint")
            save_dir = os.path.join(checkpoint_dir, str(episode_num))
            os.makedirs(save_dir)

            torch.save(agent.state_dict(), os.path.join(save_dir, "agent.pt"))


if __name__ == "__main__":
    main()
