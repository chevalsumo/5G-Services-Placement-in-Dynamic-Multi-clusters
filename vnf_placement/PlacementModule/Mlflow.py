import mlflow 
from typing import Any, Dict, Tuple, Union
import numpy as np
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
import sys 
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from typing import Dict, Any, Union, Callable, Optional
from torch import nn
from stable_baselines3 import PPO 
from vnf_placement.PlacementModule.PlacementEnv import *
from vnf_placement.PlacementModule.PlacementModule import *
from mlflow.tracking import MlflowClient
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, EveryNTimesteps, StopTrainingOnNoModelImprovement

import gym
import numpy as np
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class FineTuner:
    def __init__(self, boundaries = None, placemet_module = None,
                  mlflow_url = None,
                  mlflow_volume = None, 
                  domaines = None,
                  DEFAULT_VNF = None,
                  DEFAULT_VL = None,
                  edges_clusters = None, 
                  model_name = None) -> None:
        self._ppo_search_space, self._env_params = self.get_search_space()
        self._rl_boundaries = boundaries
        self._pm = placemet_module
        self._mlflow_url = mlflow_url
        self._model_name = model_name
        self._mlflow_volume = mlflow_volume
        
        # For parallel training (Multiple Envrionnement)
        self._domaines = domaines
        self._DEFAULT_VNF = DEFAULT_VNF
        self._DEFAULT_VL = DEFAULT_VL
        self._edges_clusters = edges_clusters
        
    def get_search_space(self):
        ppo_search_space = {
            "batch_size": hp.choice("batch_size", [64, 128, 256, 512, 1024]),
            "n_steps": hp.choice("n_steps", [64, 128, 256, 512, 1024, 2048]),
            "gamma": hp.choice("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
            "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(0.005)),
            "lr_schedule": hp.choice("lr_schedule", ['linear', 'constant']),
            "ent_coef": hp.loguniform("ent_coef", np.log(0.00000001), np.log(0.1)),
            "clip_range": hp.choice("clip_range", [0.1, 0.2, 0.4]),
            "n_epochs": hp.choice("n_epochs", [1, 5, 10, 20]),
            "gae_lambda": hp.choice("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
            "max_grad_norm": hp.choice("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]),
            "vf_coef": hp.uniform("vf_coef", 0, 1),
            "ortho_init": hp.choice("ortho_init", [False, True]),
            "activation_fn": hp.choice("activation_fn", ['tanh', 'relu', 'elu', 'leaky_relu']),
            "net_arch_width": hp.choice("net_arch_width", [64, 128, 256, 512, 1024]),
            "net_arch_depth": hp.randint("net_arch_depth", 3) + 2,
            "rew_chain_f" : hp.choice("rew_chain_f", [-50 , -10, -5]),
            "rew_st_s" : hp.choice("rew_st_s", [0, 50, 100]),   
        }

        env_params = {
        "ep" : "Request-successful",
        "lp" : 0.6,               #load pourcentage for the load reset    
        "rew_st_f" : -0,     
        "rew_chain_s" : 0,
        "rew_type" : "Best",
        "rew_rs_step_tries" : 0, 
        #"rew_st_s": ppo_search_space["rew_st_s"],
        #"rew_chain_f": ppo_search_space["rew_chain_f"],
        }

        return ppo_search_space, env_params
    
    def get_load_placement_module(self, load, clusters = None):
        print(clusters)
        return PlacementModule(self._domaines, self._rl_boundaries, self._DEFAULT_VNF, self._DEFAULT_VL, self._edges_clusters, _init_load = load, _clustering = False, _nb_cluster =4, _clusters = clusters)
    def objective(self, search_space):
        # Define Model parametrs 

        net_arch_width = search_space["net_arch_width"]
        net_arch_depth = search_space["net_arch_depth"]
        activation_fn = search_space["activation_fn"]
        net_arch = [dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
        policy = dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=search_space["ortho_init"],
            )
        
        if search_space["batch_size"] > search_space["n_steps"]:
            search_space["batch_size"] = search_space["n_steps"]

        if search_space["lr_schedule"] == "linear":
            search_space["learning_rate"] = self.linear_schedule(search_space["learning_rate"])

        ppo_parm = {
            "n_steps":search_space["n_steps"] ,
            "batch_size": search_space["batch_size"],
            "gamma": search_space["gamma"],
            "learning_rate": search_space["learning_rate"],
            "ent_coef": search_space["ent_coef"],
            "clip_range": search_space["clip_range"],
            "n_epochs": search_space["n_epochs"],
            "gae_lambda": search_space["gae_lambda"],
            "max_grad_norm": search_space["max_grad_norm"],
            "vf_coef": search_space["vf_coef"],
            "policy_kwargs": policy
        }

        mlflow.set_tracking_uri(self._mlflow_url)

        AG = AugmentedGraph(4)
        adj = self._domaines[0].build_adj_attirbutes()
        _clusters = None #AG.get_clusters(_adj_attrs = adj[:,:,1])
        print(_clusters)
        with mlflow.start_run(experiment_id= 985137944109709055) as run:
            nbenv = [0, 10 ,20, 30, 40, 50, 60]
            _envs = []

            for i in nbenv:
                print(f"ENV: {i}")
                self._env_params["lp"] = (i + 10)/100 
                load_pourcent = self._env_params["lp"]
                _episode_type = self._env_params["ep"]
                _penv_i = ClusEnv(_id_domain = 0,boundaries= self._rl_boundaries, requests= None,
                                    PModule = self.get_load_placement_module(i, clusters = _clusters), 
                                    episode_type = _episode_type, 
                                    reward_type = self._env_params["rew_type"], 
                                    _reset_load_pourcent = load_pourcent, 
                                    mask_cpu = False,
                                    mask_ram = False, 
                                    mask_bw = False, 
                                    mask_delay = True,
                                    rew_st_s = search_space["rew_st_s"] , 
                                    rew_st_f = self._env_params["rew_st_f"],
                                    rew_chain_s = self._env_params["rew_chain_s"],
                                    rew_chain_f = search_space["rew_chain_f"],
                                    rew_rs_step_tries = self._env_params["rew_rs_step_tries"],
                                    clustering = False,
                                    nb_cluster = 4,
                                    keep_clusters = False,
                                    verbose= 0,

                                )
                log_dir = None
                _penv_i = Monitor(_penv_i, log_dir, allow_early_resets=True)
            _envs.append(_penv_i)
            envs = DummyVecEnv([lambda : _envs.pop() for i in range(len(_envs))])
            loggers = Logger(
        folder=None,
        #output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        output_formats=[MLflowOutputFormat()],
        )
            stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=7,
            min_evals=6, verbose=1

                                )
            eval_callback = EvalCallback(envs, eval_freq=7000, 
                             callback_after_eval=stop_train_callback,
                             deterministic= False,
                             verbose=1)
            model_ray = PPO(
                "MultiInputPolicy", 
                envs, 
                verbose=0, 
                device= "cuda", 
                **ppo_parm,
                            )
            mlflow.log_params(self._env_params)
            mlflow.log_params({"clusters":_clusters})
            mlflow.log_params(search_space) # Not ppo_parm beacause it's not full string
            model_ray.set_logger(loggers)
            model_ray.learn(400000, progress_bar= False, callback=eval_callback)


            # Evaluate the model after training: 
            n_eval_episodes = 10
            #-1 Evaluate the model in a probabilistic maner: 
            """
            eps_rwds, eps_length, eps_success = self.evaluate_policy(model_ray, 
                envs, 
                n_eval_episodes=n_eval_episodes,
                deterministic= False,
                return_episode_rewards = True,
                return_episode_success_len= True)
            
            #-1 Save evaluation logs to mlflow DB 
            for i in range(1, n_eval_episodes +1):
                mlflow.log_metric("eval/reward", eps_rwds[i-1], i)
                mlflow.log_metric("eval/length", eps_length[i-1], i)
                mlflow.log_metric("eval/success", eps_success[i-1], i)
                
            #-1 Evaluate the model in a deterministic maner: 
 
            eps_rwds, eps_length, eps_success = self.evaluate_policy(model_ray, 
                envs, 
                n_eval_episodes=n_eval_episodes,
                deterministic= True,
                return_episode_rewards = True,
                return_episode_success_len= True)
            
            #-1 Save evaluation logs to mlflow DB 
            for i in range(1, n_eval_episodes +1):
                mlflow.log_metric("eval_det/reward", eps_rwds[i-1], i)
                mlflow.log_metric("eval_det/length", eps_length[i-1], i)
                mlflow.log_metric("eval_det/success", eps_success[i-1], i)

            """
            # Save the model with the native 
            artifact_path = f"{run.info.artifact_uri}/native_model.pkl".replace("file://", "")
            print(run.info.artifact_uri)
            print(artifact_path)
            if self._mlflow_volume is not None :
                ppo_model_path = f"{self._mlflow_volume}/{artifact_path}"
            else : 
                ppo_model_path = artifact_path
            model_ray.save(ppo_model_path)

            # Define MLflow artifact
            artifacts = {self._model_name : ppo_model_path}
            print(ppo_model_path)

            #mlflow.set_registry_uri(self._mlflow_url)
            mlflow.pyfunc.log_model(
                artifact_path = "test7",
                artifacts=artifacts, 
                python_model= PPOWrapper_()
            )      
        return {
            "loss" : 1, 'status': STATUS_OK
            #'loss': -1 * sum(eps_rwds)/len(eps_rwds), 'status': STATUS_OK
        }
    
    def linear_schedule(self, initial_value: Union[float, str]) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: (float or str)
        :return: (function)
        """
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0
            :param progress_remaining: (float)
            :return: (float)
            """
            return progress_remaining * initial_value

        return func


    def get_prod_model(self, model_name):
        client = MlflowClient(tracking_uri= self._mlflow_url)
        versions = client.get_latest_versions(name= model_name)
        prod_v = None 
        for version in versions : 
            if version.current_stage == "Production" :
                prod_v = version 
        test_model = mlflow.pyfunc.load_model(prod_v.source)

        return test_model

    def tune(self, max_trials):
        trials = Trials()
        self._best = fmin(
        self.objective,
        space=self._ppo_search_space,
        algo=tpe.suggest,
        max_evals=max_trials,  # Nombre d'évaluations à effectuer
        trials=trials,
        )   

    def evaluate_policy(
        self,
        model: "type_aliases.PolicyPredictor",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        return_episode_success_len: bool = False,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """
        Runs policy for ``n_eval_episodes`` episodes and returns average reward.
        If a vector env is passed in, this divides the episodes to evaluate onto the
        different elements of the vector env. This static division of work is done to
        remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
        details and discussion.

        .. note::
            If environment has not been wrapped with ``Monitor`` wrapper, reward and
            episode lengths are counted as it appears with ``env.step`` calls. If
            the environment contains wrappers that modify rewards or episode lengths
            (e.g. reward scaling, early episode reset), these will affect the evaluation
            results as well. You can avoid this by wrapping environment with ``Monitor``
            wrapper before anything else.

        :param model: The RL agent you want to evaluate. This can be any object
            that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
            or policy (``BasePolicy``).
        :param env: The gym environment or ``VecEnv`` environment.
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param deterministic: Whether to use deterministic or stochastic actions
        :param render: Whether to render the environment or not
        :param callback: callback function to do additional checks,
            called after each step. Gets locals() and globals() passed as parameters.
        :param reward_threshold: Minimum expected reward per episode,
            this will raise an error if the performance is not met
        :param return_episode_rewards: If True, a list of rewards and episode lengths
            per episode will be returned instead of the mean.
        :param warn: If True (default), warns user about lack of a Monitor wrapper in the
            evaluation environment.
        :return: Mean reward per episode, std of reward per episode.
            Returns ([float], [int]) when ``return_episode_rewards`` is True, first
            list containing per-episode rewards and second containing per-episode lengths
            (in number of steps).
        """
        is_monitor_wrapped = False
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor

        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

        if not is_monitor_wrapped and warn:
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                "Consider wrapping environment first with ``Monitor`` wrapper.",
                UserWarning,
            )

        n_envs = env.num_envs
        episode_rewards = []
        episode_lengths = []
        episode_success = []

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")

        current_success = np.zeros(n_envs, dtype="int")

        observations = env.reset()
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
            observations, rewards, dones, infos = env.step(actions)
            current_rewards += rewards
            current_lengths += 1
            current_success += (rewards > 1)

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:

                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if callback is not None:
                        callback(locals(), globals())

                    if dones[i]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_success.append(current_success[i])

                            episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        current_success[i] = 0

            if render:
                env.render()

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        if return_episode_rewards and return_episode_success_len:
            return episode_rewards, episode_lengths, episode_success
        elif return_episode_rewards:
            return episode_rewards, episode_lengths
        return mean_reward, std_reward
    
class PPOWrapper_(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        from stable_baselines3 import PPO 
        print("hna")
        self.ppo_model = PPO.load(context.artifacts["salim"])

    def predict(self, context, model_input):
        return int(self.ppo_model.predict(model_input, deterministic=False)[0])


