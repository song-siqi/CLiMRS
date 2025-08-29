import time
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from coohoi.env.tasks.humanoid_amp_carryobject_obstacle import HumanoidAMPCarryObjectObstacle


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CoohoiAdapter:
    """
    通用适配器：把 coohoi 仿真任务适配为 LLM 框架期望的环境接口。

    初始化方式：
    - 直接传入已构建的 task 实例：CoohoiAdapter(task=task_instance, sim_step_fn=sim_step_fn)
    - 或传入一个 factory：CoohoiAdapter(task_factory=lambda: create_task(...), sim_step_fn=sim_step_fn)

    必需/可选参数：
    - task: 已创建的仿真任务实例（优先）
    - task_factory: 可调用以创建任务实例的函数（在 task 为空时使用）
    - sim_step_fn: 可选，可调用以推进物理仿真一帧（例如 simulate.step()）。如果未提供，适配器会尝试调用 task 的公开推进接口（如 task.sim_step 或 task.step_sim 等）。

    提供的方法：reset(), get_observations(), step(high_level_action), close()
    """

    def __init__(
        self,
        task: Any = None,
        task_factory: Optional[Callable[[], Any]] = None,
        sim_step_fn: Optional[Callable[[], None]] = None,
    ):
        self.task = task
        self._task_factory = task_factory
        self._sim_step_fn = sim_step_fn
        self.current_step = 0
        self._init_task_if_needed()

    def _init_task_if_needed(self):
        if self.task is None and self._task_factory is not None:
            try:
                self.task = self._task_factory()
                logger.info("CoohoiAdapter: task created from factory")
            except Exception as e:
                logger.exception("CoohoiAdapter: failed to create task from factory: %s", e)
        if self.task is None:
            logger.warning("CoohoiAdapter: no task instance available. Adapter will operate in stub mode.")

    def reset(self) -> Dict:
        """重置环境并返回初始观测（LLM 期望格式的字典）"""
        self.current_step = 0
        if self.task is not None:
            # 尝试调用常见的 reset 接口
            if hasattr(self.task, 'reset'):
                try:
                    self.task.reset()
                except TypeError:
                    try:
                        self.task.reset([])
                    except Exception:
                        logger.debug("task.reset() failed with unknown signature")
            # 兼容 _reset_task
            if hasattr(self.task, '_reset_task'):
                try:
                    # 如果该方法需要 env_ids 列表，传入默认值
                    self.task._reset_task([0])
                except Exception:
                    try:
                        self.task._reset_task(0)
                    except Exception:
                        logger.debug("task._reset_task() call failed or not applicable")
        obs = self.get_observations()
        return obs

    def get_observations(self) -> Dict:
        """返回一个 LLM 可用的观测字典。优先使用 task 提供的观测接口，否则返回简化状态。"""
        if self.task is None:
            return {'stub': True, 'step': self.current_step}

        # 优先使用 task 中类似 Get_env_info 的接口
        if hasattr(self.task, 'get_observations'):
            try:
                return self.task.get_observations()
            except Exception:
                logger.debug("task.get_observations() raised exception, fallback to _build_state")

        # 尝试使用 get_observation(agent_id)
        if hasattr(self.task, 'get_observation'):
            try:
                return self.task.get_observation(0)
            except Exception:
                logger.debug("task.get_observation(agent_id) failed")

        # 回退：从常见字段构建观测
        obs = {'step': self.current_step}
        if hasattr(self.task, '_box_pos'):
            try:
                box_pos = getattr(self.task, '_box_pos')
                obs['box_pos'] = box_pos.cpu().numpy().tolist() if hasattr(box_pos, 'cpu') else box_pos.tolist()
            except Exception:
                pass
        if hasattr(self.task, '_target_pos'):
            try:
                tar = getattr(self.task, '_target_pos')
                obs['target_pos'] = tar.cpu().numpy().tolist() if hasattr(tar, 'cpu') else tar.tolist()
            except Exception:
                pass
        # 添加任务图（如果有）以保持和 LLM 框架一致
        if hasattr(self.task, 'graph'):
            obs['graph'] = getattr(self.task, 'graph')
        return obs

    def step(self, high_level_action: Any, sim_steps: int = 20) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一个高层动作并推进若干仿真步。
        high_level_action 可以是：
          - 字符串，例如: "move_to 3.0 4.0"、"grasp 12"、"place 3.0 4.0"
          - dict，例如 {'action':'move_to','target':[x,y]}

        返回 (obs, reward, done, info)
        """
        self.current_step += 1
        action = self._normalize_action(high_level_action)
        # 将高层动作翻译为 task 可调用的方法或低层 control 指令
        sim_cmd = self._translate_action(action)

        # 如果 task 提供了高层执行接口，优先调用
        executed = False
        info: Dict = {}

        if self.task is not None:
            # 如果 task 支持 apply_high_level_action 或类似名称，调用它
            for attr in ('apply_high_level_action', 'apply_action', 'step_action', 'do_action'):
                if hasattr(self.task, attr):
                    try:
                        getattr(self.task, attr)(action)
                        executed = True
                        break
                    except Exception:
                        logger.debug("task.%s failed for action %s", attr, action)

            # 作为替代：如果翻译为 control（低层），则把 control 传给 task.pre_physics_step
            if not executed and sim_cmd.get('control') is not None and hasattr(self.task, 'pre_physics_step'):
                control = sim_cmd['control']
                # 多步推进
                for _ in range(sim_steps):
                    try:
                        # pre_physics_step 通常期望 actions，传入 control
                        self.task.pre_physics_step(control)
                    except Exception:
                        # 如果不匹配签名，尝试以单参数调用
                        try:
                            self.task.pre_physics_step()
                        except Exception:
                            logger.debug('task.pre_physics_step invocation failed')
                    self._advance_simulation_once()
                executed = True

        # 如果仍未被 task 执行，则由适配器模拟改变（stub 模式）
        if not executed:
            logger.info('CoohoiAdapter: executing stub for action: %s', action)
            # 简单模拟：记录 info 并推进 sim_steps
            for _ in range(sim_steps):
                self._advance_simulation_once()

        obs = self.get_observations()
        reward = self._compute_reward(obs, action)
        done = self._check_done(obs)
        info.update({'executed_by_task': executed})
        return obs, float(reward), bool(done), info

    def _advance_simulation_once(self):
        """推进一帧物理仿真：优先使用外部 sim_step_fn，其次尝试 task 暴露的 step 接口。"""
        if self._sim_step_fn is not None:
            try:
                self._sim_step_fn()
                return
            except Exception:
                logger.debug("sim_step_fn failed")
        if self.task is not None:
            for name in ('step_sim', 'step', 'simulate', 'sim_step', 'step_world'):
                if hasattr(self.task, name):
                    try:
                        getattr(self.task, name)()
                        return
                    except Exception:
                        logger.debug('task.%s() failed while advancing simulation', name)
        # 否则睡眠以模拟时间推进
        time.sleep(0.01)

    def _normalize_action(self, action: Any) -> Dict:
        """把多种类型的 high_level_action 标准化为 dict 格式。"""
        if isinstance(action, dict):
            return action
        if isinstance(action, str):
            toks = action.strip().split()
            if len(toks) == 0:
                return {'action': 'noop'}
            verb = toks[0].lower()
            if verb == 'move_to' or verb == 'move' or verb == 'moveto':
                try:
                    x = float(toks[1]); y = float(toks[2])
                    return {'action': 'move_to', 'target': [x, y]}
                except Exception:
                    return {'action': 'move_to', 'target': None}
            if verb in ('grasp', 'grab', 'pick'):
                try:
                    obj = int(toks[1])
                    return {'action': 'grasp', 'object_id': obj}
                except Exception:
                    return {'action': 'grasp', 'object_id': toks[1] if len(toks) > 1 else None}
            if verb in ('place', 'put', 'release'):
                try:
                    x = float(toks[1]); y = float(toks[2])
                    return {'action': 'place', 'target': [x, y]}
                except Exception:
                    return {'action': 'place', 'target': None}
            # 默认：把整条字符串当作命令
            return {'action': verb, 'raw': ' '.join(toks[1:])}
        # 其他类型
        return {'action': 'noop'}

    def _translate_action(self, action: Dict) -> Dict:
        """把标准化后的高层动作翻译为低层 control 或 task 可调用的参数。
        返回字典：{'control': ..., 'num_sim_steps': n}
        如果适配到 task 的高层方法（例如 plan_franka_path_to_grasp）也可以把该方法名放入返回的 'method' 字段，供 step() 直接调用。
        """
        if not isinstance(action, dict):
            return {'control': None, 'num_sim_steps': 20}

        cmd = action.get('action')
        if cmd == 'move_to':
            # 试图调用 task 的移动/导航接口
            if self.task is not None:
                for method in ('_reset_target', '_reset_car_target', '_reset_car_target2', 'move_to', '_move_to_waypoint'):
                    if hasattr(self.task, method):
                        return {'method': method, 'args': (action.get('target'),), 'num_sim_steps': 60}
            # 否则返回空 control
            return {'control': {'type': 'move_to', 'target': action.get('target')}, 'num_sim_steps': 60}

        if cmd == 'grasp':
            # 试图调用 franka 抓取的高层规划接口
            if self.task is not None:
                for method in ('plan_franka_path_to_grasp', 'plan_franka_path_to_pre_grasp', 'plan_franka_path_to_place', 'close_gripper'):
                    if hasattr(self.task, method):
                        # prefer grasp planning first
                        if method == 'plan_franka_path_to_grasp':
                            return {'method': method, 'args': (), 'num_sim_steps': 120}
                # fallback: call close_gripper
                if hasattr(self.task, 'close_gripper'):
                    return {'method': 'close_gripper', 'args': (), 'num_sim_steps': 40}
            return {'control': {'type': 'grasp', 'object_id': action.get('object_id')}, 'num_sim_steps': 80}

        if cmd == 'place':
            if self.task is not None:
                if hasattr(self.task, 'plan_franka_path_to_place'):
                    return {'method': 'plan_franka_path_to_place', 'args': (), 'num_sim_steps': 120}
            return {'control': {'type': 'place', 'target': action.get('target')}, 'num_sim_steps': 80}

        if cmd == 'noop':
            return {'control': None, 'num_sim_steps': 1}

        # 默认：多步推进保持空控制
        return {'control': None, 'num_sim_steps': 20}

    def _compute_reward(self, obs: Dict, action: Dict) -> float:
        """计算简化 reward，优先使用 task 的奖励计算接口（如果存在）。"""
        if self.task is not None:
            # 如果 task 有 _compute_reward 接口并且可调用
            if hasattr(self.task, '_compute_reward'):
                try:
                    # 有些实现需要 actions 参数
                    return float(self.task._compute_reward(action))
                except Exception:
                    try:
                        return float(self.task._compute_reward())
                    except Exception:
                        logger.debug("task._compute_reward failed")
        # 简单 heuristic：每步微小负奖励
        return -0.01

    def _check_done(self, obs: Dict) -> bool:
        """检查是否结束：优先使用 task 的终止判断。"""
        if self.task is not None:
            if hasattr(self.task, '_compute_reset'):
                try:
                    # 该函数通常会填充 reset_buf 或返回布尔数组
                    # 我们尝试以布尔返回值解释
                    res = self.task._compute_reset()
                    return bool(res)
                except Exception:
                    logger.debug('task._compute_reset invocation failed')
            if hasattr(self.task, '_check_done'):
                try:
                    return bool(self.task._check_done())
                except Exception:
                    pass
        # 默认不结束
        return False

    def close(self):
        """释放/关闭仿真资源（如果 task 提供 close/cleanup 接口则调用）。"""
        if self.task is not None:
            for name in ('close', 'cleanup', 'destroy'):
                if hasattr(self.task, name):
                    try:
                        getattr(self.task, name)()
                    except Exception:
                        logger.debug('task.%s() failed during close', name)
        logger.info('CoohoiAdapter closed')