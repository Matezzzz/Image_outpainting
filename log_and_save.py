import argparse
import os
import json
#import re

import numpy as np
import tensorflow as tf
import wandb



class EvaluateFunc:
    name : str
    metric_name : str
    func : callable

    def __init__(self, name, func):
        self.name = name
        self.metric_name = f"evaluate_{name}"
        self.func = func


WANDB_USER_DEFAULT = "matezzzz"
class WandbManager:
    user : str
    project : str
    resumed_run : object


    def __init__(self, project, user = WANDB_USER_DEFAULT):
        self.user = user
        self.project = project
        self.resumed_run = None
        self.resume_args = None

    def start(self, args):
        self.resumed_run = None
        self.resume_args = args
        wandb.init(project=self.project, config=args)
        wandb.run.log_code(".")
        return wandb.run.name

    def resume(self, resume_name_part, args):
        api = wandb.Api()
        runs = api.runs(f"{self.user}/{self.project}", filters={"display_name": {"$regex":f".*{resume_name_part}.*"}})
        self.resumed_run = runs[0]
        if self.resumed_run.state == "running":
            raise ValueError(f"Cannot resume an active run. Filter \'{resume_name_part}\'")

        self.resume_args = vars(args)
        j = json.loads(self.resumed_run.json_config)
        for key, val in j.items():
            if key in self.resume_args:
                try:
                    self.resume_args[key] = val['value']
                except KeyError:
                    print (f"Failed to load value: '{key}': {val}")
        self.resume_args = argparse.Namespace(**self.resume_args)
        wandb.init(project=self.project, config=self.resume_args, resume="must", id=self.resumed_run.id)
        wandb.run.log_code(".")

    @property
    def resumed(self):
        return self.resumed_run is not None

    def scan_history(self, pattern):
        if not self.resumed:
            print ("ERROR: Scanning history for a run that was not resumed")
        return list(self.resumed_run.scan_history(pattern))


class Log:
    log : dict
    def __init__(self) -> None:
        self.log = {}

    def log_image(self, name, image):
        self.log[name] = wandb.Image(image)
        return self

    def log_images(self, name, images):
        self.log[name] = [wandb.Image(i) for i in images]
        return self

    def log_value(self, name, value):
        self.log[name] = value
        return self

    def log_dict(self, dictionary):
        self.log |= dictionary
        return self

    def commit(self):
        wandb.log(self.log)
        self.log = {}


class TrainingLog(tf.keras.callbacks.Callback):
    def __init__(self, dev_dataset, test_dataset, log_frequency, test_frequency, validation_frequency=None, train_batch_size = None):
        super().__init__()
        self.dev_dataset = dev_dataset
        self.test_dataset = iter(test_dataset.repeat(None))
        self.log = Log()
        self.log_frequency = log_frequency
        self.test_frequency = test_frequency
        self.val_frequency = validation_frequency if validation_frequency is not None else test_frequency
        self.batch_size = dev_dataset.element_spec.shape[0] if train_batch_size is None else train_batch_size
        self.batches_processed = 0

    def on_batch_end(self, batch, logs=None):
        if self.batches_processed % self.test_frequency == 0:
            data = next(self.test_dataset)
            self.run_test(data)
        if self.batches_processed % self.val_frequency == 0:
            self.log.log_dict({("val_"+name):val for name, val in self.run_validation().items()})
        if self.batches_processed % self.log_frequency == 0:
            self.log.log_value("images_processed", self.batches_processed*self.batch_size)
            if logs is not None:
                self.log.log_dict(logs)
            self.log.commit()
        self.batches_processed += 1
        return super().on_batch_end(batch, logs)

    def run_test(self, data):
        pass

    def run_validation(self):
        return self.model.evaluate(self.dev_dataset, return_dict=True, verbose=0)








class ModelSave:
    eval_functions : list[EvaluateFunc]
    save_func : callable

    eval_episodes : list

    log_each : int
    evaluate_each : int
    save_each : int

    improvement_step : float
    perfecting_threshold : float

    best_returns : dict

    training_ep : int

    training : bool

    state_json : dict

    wandb_manager : WandbManager

    def __init__(self, project : str, args : argparse.Namespace, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part = None):

        self.eval_episodes = eval_episodes if isinstance(eval_episodes, list) else [eval_episodes]

        self.log_each = log_each
        self.evaluate_each = evaluate_each
        self.save_each = save_each

        self.improvement_step = improvement_step
        self.perfecting_threshold = perfecting_threshold

        self.training = True

        if resume_name_part is None:
            wandb.init(project=project, config=args)
        else:
            self.wandb_manager.resume(resume_name_part, args)


        self.stop_file_name = f"stop_{wandb.run.name}.txt"
        with open(self.stop_file_name, 'w', encoding="utf") as file:
            file.write("N")


    def initialize(self, evaluate_functions, save_func, load_func):
        self.eval_functions = None if evaluate_functions is None else ([EvaluateFunc("return", evaluate_functions)] if not isinstance(evaluate_functions, dict) else [EvaluateFunc(name, func) for name, func in evaluate_functions.items()])
        if not self.wandb_manager.resumed:
            self.training_ep = 0
            self.best_returns = {e.name:-float("inf") for e in self.eval_functions}
        else:
            _ = load_func
            raise NotImplementedError("Resuming not implemented!")

            # history = list(self.resumed_run.scan_history([e.metric_name for e in self.eval_functions]))
            # self.best_returns = {}
            # for e in self.eval_functions:
            #     vals = [h[e.metric_name] for h in history]
            #     self.best_returns[e.name] = max(vals) if len(vals) > 0 else -float("inf")
            # self.training_ep = len(list(self.resumed_run.scan_history(["episode"])))

            # available_models = list(self.resumed_run.files())
            # newest_model = None
            # newest_ep = 0
            # pattern = re.compile(".*_([0-9]+).h5")
            # for f in available_models:
            #     m = re.match(pattern, f.name)
            #     if m:
            #         ep = int(m[1])
            #         if ep > newest_ep:
            #             newest_model = f
            #             newest_ep = ep
            # if newest_model is not None:
            #     newest_model.download(".")
            #     load_func(".", str(newest_ep))
            #     os.unlink(newest_model.name)
            # else:
            #     print ("ERROR: Resuming, but no model could be loaded.")
        self.save_func = save_func



    def _train_step(self, log):
        log["episode"] = self.training_ep
        self.training_ep += 1
        if self.training_ep % self.log_each == 0:
            log_dict = self._train_log()
            print (f"Training step {self.training_ep}{'' if len(log_dict)==0 else f', {log_dict}'}")
        if self.training_ep % self.evaluate_each == 0:
            vals = self.evaluate()
            for name, val in vals.items():
                log[f"evaluate_{name}"] = val
        if self.training_ep % self.save_each == 0:
            self.save_func(wandb.run.dir, f"{self.training_ep}")

        wandb.log(log)

    def train_step_base(self):
        self._train_step({})

    def _train_log(self):
        return ""

    def evaluate(self):
        prev_eps = 0
        returns = []
        self.training = open(self.stop_file_name, encoding="utf").read() != "Y"
        if not self.training:
            os.unlink(self.stop_file_name)

        results = {}
        for func in self.eval_functions:
            failed = False
            print (f"{func.name}, ", end="")
            for eps in self.eval_episodes:
                for _ in range(eps - prev_eps):
                    returns.append(func.func())
                avg_ret = np.mean(returns)
                if avg_ret < self.best_returns[func.name] + self.improvement_step:
                    print (f"{eps} test: Failed, Got return: {avg_ret:.2f}, attempted to beat {self.best_returns[func.name]:.2f} by {self.improvement_step:.2f}")
                    results[func.name] = avg_ret
                    failed = True
                    break
                print (f"{eps} test: Passed, ", end="")
                if avg_ret < self.perfecting_threshold:
                    break
                prev_eps = eps
            if not failed:
                self.best_returns[func.name] = avg_ret
                print (f"New best return {avg_ret:.2f} +-{np.std(returns):.2f}")
                self.save_func(wandb.run.dir, f"{func.name}_best")
                results[func.name] = avg_ret
        return results






class EnvModelSave(ModelSave):
    training_returns : list
    train_ep_return : float

    def __init__(self, project : str, args : argparse.Namespace, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part=None):
        self.train_ep_return = 0
        self.training_returns = []
        super().__init__(project, args, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part)

    def train_step_env(self, reward, done):
        self.train_ep_return += reward
        if done:
            log = {"return":self.train_ep_return}
            self.training_returns.append(self.train_ep_return)
            self.train_ep_return = 0
            self._train_step(log)

    def _train_log(self):
        return f"Mean {self.log_each} episode return {np.mean(self.training_returns[-self.log_each:]):.2f} +-{np.std(self.training_returns[-self.log_each:]):.2f}, Mean {5*self.log_each} episode return {np.mean(self.training_returns[-5*self.log_each:]):.2f} +-{np.std(self.training_returns[-5*self.log_each:]):.2f}"



class VectorEnvModelSave(ModelSave):
    training_returns : list
    train_ep_returns : list

    def __init__(self, project : str, args : argparse.Namespace, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part=None):
        self.train_ep_returns = [0.0 for _ in range(args.envs)]
        self.training_returns = []
        super().__init__(project, args, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part)

    def train_step_vec(self, rewards, dones):
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.train_ep_returns[i] += reward
            if done:
                log = {"return":self.train_ep_returns[i]}
                self.training_returns.append(self.train_ep_returns[i])
                self.train_ep_returns[i] = 0
                self._train_step(log)

    def _train_log(self):
        return f"Mean {self.log_each} episode return {np.mean(self.training_returns[-self.log_each:]):.2f} +-{np.std(self.training_returns[-self.log_each:]):.2f}, Mean {5*self.log_each} episode return {np.mean(self.training_returns[-5*self.log_each:]):.2f} +-{np.std(self.training_returns[-5*self.log_each:]):.2f}"




# class ModelSave(ModelSave):
#     def __init__(self, project : str, args : argparse.Namespace, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part=None):
#         super().__init__(project, args, eval_episodes, log_each, evaluate_each, save_each, improvement_step, perfecting_threshold, resume_name_part)
