from env.tasks.humanoid import Humanoid
from env.tasks.humanoid_amp import HumanoidAMP
from env.tasks.humanoid_view_motion import HumanoidViewMotion
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper

# Tasks for CooHOI
from env.tasks.humanoid_amp_carryobject import HumanoidAMPCarryObject
from env.tasks.share_humanoid_amp_carryobject import ShareHumanoidCarryObject
from coohoi.env.tasks.humanoid_amp_carryobject_obstacle_0901 import HumanoidAMPCarryObjectObstacle

from isaacgym import rlgpu
from env.LLM_API.llm_observer import LLMObserver
import json
import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")

def parse_task(args, cfg, cfg_train, sim_params, is_ask_llm=True):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
        if is_ask_llm:
            llm_observer = LLMObserver()
            llm_observer.set_task(task)
            answer = llm_observer.update()
            print("LLM answer:", answer)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
