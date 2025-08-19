import os
import re
import yaml
import argparse
from examples.maintenance_scheduling_multidiscrete.environment.environment import Environment
from copy import deepcopy
from sb3.misc import make_sb3_env, linear_schedule, AutoSave, StartingSteps, CustomMetrics

from stable_baselines3 import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./examples/maintenance_scheduling_multidiscrete/config.yaml", help='Type of control policy')
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as file:
        train_config_in = yaml.safe_load(file)

    train_config = deepcopy(train_config_in)
    num_envs = train_config["num_envs"]

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(local_path, "runs", train_config["name"])
    model_folder = os.path.join(results_folder, "model")
    tensor_board_folder = os.path.join(results_folder, "tb")
    monitor_folder = os.path.join(results_folder, "monitor")

    model_save_config = train_config["model_save"]
    training_stop_config = train_config["training_stop"]
    policy_kwargs = train_config["policy_kwargs"]

    os.makedirs(model_folder, exist_ok=True)

    env = make_sb3_env(Environment, num_envs, seed=train_config["seed"], monitor_folder=monitor_folder)
    print("Activated {} environment(s)".format(num_envs))

    # Generic algo settings
    gamma = train_config["gamma"]
    model_checkpoint_path = train_config["model_checkpoint_path"]
    learning_rate = linear_schedule(train_config["learning_rate"][0], train_config["learning_rate"][1])

    clip_range = linear_schedule(train_config["clip_range"][0], train_config["clip_range"][1])
    clip_range_vf = clip_range
    n_epochs = train_config["n_epochs"]
    n_steps = train_config["n_steps"]
    batch_size = train_config["batch_size"]
    min_steps = num_envs * n_steps
    assert training_stop_config["max_time_steps"] > min_steps, "The minimum number of training steps is {}".format(min_steps)
    gae_lambda = train_config["gae_lambda"]
    normalize_advantage = train_config["normalize_advantage"]
    ent_coef = train_config["ent_coef"]
    vf_coef = train_config["vf_coef"]
    max_grad_norm = train_config["max_grad_norm"]
    use_sde = train_config["use_sde"]
    sde_sample_freq = train_config["sde_sample_freq"]
    target_kl = train_config["target_kl"]

    starting_steps = 0
    reset_num_timesteps = True
    callbacks = [CustomMetrics()]

    if model_checkpoint_path is None:
        agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                    gamma=gamma, batch_size=batch_size, n_epochs=n_epochs, n_steps=n_steps,
                    learning_rate=learning_rate, clip_range=clip_range, clip_range_vf=clip_range_vf,
                    gae_lambda=gae_lambda, normalize_advantage=normalize_advantage,
                    ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    target_kl=target_kl, tensorboard_log=tensor_board_folder, device="cuda")
    else:
        # Load the trained agent
        # Use regex to find the number after the latest underscore
        match = re.search(r'_(\d+)(?!.*_\d)', model_checkpoint_path)
        if not match:
            raise Exception(f"{model_checkpoint_path} should contain a number at the end of the filename indicating the number of training steps.")
        starting_steps = int(match.group(1))  # Convert the found number to an integer

        agent = PPO.load(model_checkpoint_path, env=env, policy_kwargs=policy_kwargs,
                         batch_size=batch_size, n_epochs=n_epochs, n_steps=n_steps, gamma=gamma,
                         learning_rate=learning_rate, clip_range=clip_range, clip_range_vf=clip_range_vf,
                         gae_lambda=gae_lambda, normalize_advantage=normalize_advantage, ent_coef=ent_coef,
                         vf_coef=vf_coef, max_grad_norm=max_grad_norm, target_kl=target_kl,
                         tensorboard_log=tensor_board_folder, device="cuda")
        reset_num_timesteps = False
        callbacks.append(StartingSteps(starting_steps=starting_steps))

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave = model_save_config["autosave"]["active"]
    autosave_freq = model_save_config["autosave"]["frequency"]

    if autosave:
        callbacks.append(AutoSave(check_freq=autosave_freq, num_envs=num_envs, save_path=model_folder, filename_prefix="model_", starting_steps=starting_steps))

    # Train the agent
    time_steps = training_stop_config["max_time_steps"]
    agent.learn(total_timesteps=time_steps, reset_num_timesteps=reset_num_timesteps, callback=callbacks)

    # Save the agent
    new_model_checkpoint = "model_" + str(starting_steps + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Free memory
    assert agent.env is not None
    agent.env.close()
    del agent.env
    del agent