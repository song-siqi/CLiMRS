import requests
import json
import re
import os
from typing import Dict, Tuple

try:
    from isaacgym import gymtorch
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False
    gymtorch = None

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from LLM.dev_revision.llm_agents.feedback_agent import FeedbackAgent
    from LLM.dev_revision.llm_agents.oracle_planner import OraclePlanner
    ADVANCED_LLM_AVAILABLE = True
except ImportError as e:
    ADVANCED_LLM_AVAILABLE = False

class LLMWorkflow:
    def __init__(self, area_positions: Dict[str, Tuple[float, float, float]], 
                 agent_positions: Dict[str, Tuple[float, float]], 
                 actor_indices_map: Dict[str, int], 
                 use_oracle: bool = True):
        self.area_positions = area_positions
        self.agent_positions = agent_positions
        self.actor_indices_map = actor_indices_map
        self.API_URL = "http://35.220.164.252:3888/v1/chat/completions"  
        self.API_KEY = "sk-VS6OzWyx7SyeeNnWRo7BuUeD9H9jzxU88z9IQlcf4K72l14U"
        self.use_oracle = use_oracle and ADVANCED_LLM_AVAILABLE
        self.oracle_planner = None
        
        if self.use_oracle:
            self._initialize_oracle_planner()
    
    def _initialize_oracle_planner(self):
        try:
            from types import SimpleNamespace
            
            args = SimpleNamespace()
            args.debug = False
            args.source = 'llm_module'
            args.lm_id = 'gpt-4o-mini'
            args.max_tokens = 1000
            args.t = 0.7
            args.n = 1
            args.env = 'env0'
            args.api_key = None
            args.organization = None
            args.oracle_prompt_path = 'LLM/dev_revision/prompt/oracle_prompt.txt'
            args.agent_selection_prompt_path = 'LLM/dev_revision/prompt/agent_selection_prompt.txt'
            
            def env_fn():
                return None
                
            def agent_fn():
                return []
            
            self.oracle_planner = OraclePlanner(
                environment_fn=env_fn,
                agent_fn=[],  # 修复：传递空列表而不是函数
                args=args,
                run_predefined_actions=False,
                oracle_prompt_path=args.oracle_prompt_path,
                agent_selection_prompt_path=args.agent_selection_prompt_path,
            )
        except Exception as e:
            print(f"Failed to initialize oracle planner: {e}")
            self.use_oracle = False
        
    def ask_llm(self, question):
        if self.use_oracle and self.oracle_planner:
            return self._ask_oracle_llm(question)
        else:
            return self._ask_fallback_llm(question)

    def _ask_oracle_llm(self, question):
        try:
            obs_text = self._format_oracle_observation_to_text()
            
            print(f"🔮 使用Oracle规划系统，观察文本: {obs_text[:200]}...")
            
            vanilla_message, usage = self.oracle_planner.oracle_planning_vanilla(
                obs_text=obs_text,
                goal_instruction=question,
                num_agents=5,
                dialogue_history="",
            )
            
            print(f"🔮 Oracle原始响应: {vanilla_message}")
            
            message, usage = self.oracle_planner.extract_structured_message(vanilla_message)
            
            print(f"🔮 Oracle结构化响应: {message}")
            return message
            
        except Exception as e:
            print(f"❌ Oracle LLM失败，降级到fallback: {e}")
            import traceback
            traceback.print_exc()
            return self._ask_fallback_llm(question)

    def _format_observation_to_text(self):
        obs_text = ""
        
        if self.area_positions:
            obs_text += "Areas and Components:\n"
            for area, pos in self.area_positions.items():
                obs_text += f"Area {area}: Component at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
        
        if self.agent_positions:
            obs_text += "\nAgent Positions:\n"
            for agent, pos in self.agent_positions.items():
                obs_text += f"{agent}: Position ({pos[0]:.2f}, {pos[1]:.2f})\n"
        
        obs_text += "\nCurrent Task: Assembly of robot components with humanoid, mobile robots, and franka arm coordination.\n"
        
        return obs_text

    def _format_oracle_observation_to_text(self):
        obs_text = ""
        
        # 按照dev_revision/arena.py中agent_obs2text的格式
        if self.agent_positions:
            for agent_name, pos in self.agent_positions.items():
                if "humanoid" in agent_name.lower():
                    obs_text += f"I am <humanoid>(101). Now my state is: STANDING. I am INSIDE the <assembly room>(1).\n"
                elif "franka" in agent_name.lower():
                    obs_text += f"I am <franka>(606). Now my state is: READY. I am ON the <high table>(2).\n"
                elif "mobile_car" in agent_name.lower() or "wheeled robot" in agent_name.lower():
                    obs_text += f"I am <mobile_car>(202). Now my state is: IDLE. I am INSIDE the <assembly room>(1).\n"
        
        obs_text += "\nNow I am in the <assembly room>(1). In this room, I can see:\n"
        
        if self.area_positions:
            for area, pos in self.area_positions.items():
                if area == 'A':
                    obs_text += f"<trunk>(303). Its properties are: ASSEMBLABLE. Now its state is: ON_SURFACE.\n"
                elif area == 'B':
                    obs_text += f"<left wheel>(405). Its properties are: GRABABLE. Now its state is: ON_SURFACE.\n"
                elif area == 'C':
                    obs_text += f"<right wheel>(406). Its properties are: GRABABLE. Now its state is: ON_SURFACE.\n"
                elif area == 'D':
                    obs_text += f"<obstacles>(507). Its properties are: CARRIABLE. Now its state is: ON_SURFACE.\n"
        
        
        obs_text += "\nThese objects have a certain position relationship with each other:\n"
        if self.area_positions:
            for area, pos in self.area_positions.items():
                if area == 'A':
                    obs_text += f"The <trunk>(303) is ON the <surface_A>(200).\n"
                elif area == 'B':
                    obs_text += f"The <left wheel>(405) is ON the <surface_B>(201).\n"
                elif area == 'C':
                    obs_text += f"The <right wheel>(406) is ON the <surface_C>(202).\n"
                elif area == 'D':
                    obs_text += f"The <obstacles>(507) is ON the <surface_D>(203).\n"
        
        obs_text += "The franka assembly area is at the <assembly table>(2).\n"
        
        return obs_text

    def _ask_fallback_llm(self, question):
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.API_KEY}',
            'User-Agent':'Apifox/1.0.0(https://apifox.com)',
            'Content-Type': 'application/json'
        }
        # 使用与Oracle prompt对齐的格式
        prompt = """{}
                    Suppose you think of yourself as a robotics assembly helper named Oracle. There are different types of robot agents in the workspace, each with different abilities and action spaces. When executing a task, if the capabilities and action space of one of the agents are insufficient to complete the current instructions, other agents with different abilities will be needed to assist and cooperate to complete a robot assembly task.

                    The robot agents and their available skills are:

                    Humanoid (101): Available skills:
                    - [walk] <humanoid> (101) move to selected area
                    - [carry] <humanoid> (101) carry <obstacles> (507)
                    - [wait] <humanoid> (101) wait

                    Mobile Car (202): Available skills:
                    - [move] <mobile_car> (202) move to component location using RRT path
                    - [push] <mobile_car> (202) push selected component to franka area
                    - [wait] <mobile_car> (202) wait

                    Franka Robot Arm (606): Available skills:
                    - [check] <franka> (606) check <trunk> (303)
                    - [check] <franka> (606) check <left wheel> (405)
                    - [check] <franka> (606) check <right wheel> (406)
                    - [pick] <franka> (606) pick and place <left wheel> (405) on <trunk> (303)
                    - [pick] <franka> (606) pick and place <right wheel> (406) on <trunk> (303)
                    - [wait] <franka> (606) wait

                    Observation System: Available skills:
                    - [observe] area <A> (001) - only observe, robot doesn't move
                    - [observe] area <B> (002) - only observe, robot doesn't move
                    - [observe] area <C> (003) - only observe, robot doesn't move
                    - [observe] area <D> (004) - only observe, robot doesn't move

                    The goal of the task is: {}

                    Current Environment Observation:
                    {}

                    Assembly Task Workflow:
                    1. Use [observe] skills to locate components in areas A, B, C, D
                    2. Use wheeled robots to [move] to component locations and [push] components to franka area
                    3. Use humanoid to [carry] obstacles (507) if path clearing is needed
                    4. Use franka to [check] components and [pick] and place wheels on trunk for final assembly
                    5. Coordinate between all agents using [wait] skills when needed

                    Please provide the most critical next action using the specific skill format. The output should be in the format: "Hello <robot_type>(id): [skill_name] instruction."

                    Answer: Let's think step by step.
                """

        formatted_question = self.build_prompt_with_positions(question)
        prompt_filled = prompt.format(question, question, formatted_question)
        prompt_filled = self.replace_placeholders_with_positions(prompt_filled)
        

        payload = json.dumps({
            "model":"gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt_filled}
            ]
        })

        try:
            response = requests.post(self.API_URL, headers=headers, data=payload)
            response_json = response.json()
            content = response_json.get("choices", [])[0].get("message", {}).get("content", "")
            return content
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return f"Error: {e}"

    def build_prompt_with_positions(self, question: str) -> str:
        env_obs = ""
        print(f"🔍 LLM环境观察 - area_positions: {self.area_positions}")
        print(f"🔍 LLM环境观察 - agent_positions: {self.agent_positions}")
        
        if self.area_positions:
            env_obs += "Components found in areas:\n"
            for k, v in self.area_positions.items():
                if v is None or any(x is None for x in v):
                    continue
                env_obs += f"- Area <{k}> ({100 + ord(k)}): component at ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})\n"
        else:
            env_obs += "No components detected in any area yet.\n"
            
        if self.agent_positions:
            env_obs += "Current agent positions:\n"
            for name, pos in self.agent_positions.items():
                env_obs += f"- {name} at ({pos[0]:.2f}, {pos[1]:.2f})\n"
                
        if self.area_positions:
            areas_with_components = list(self.area_positions.keys())
            env_obs += f"\n🚨 CRITICAL STRATEGY: ONLY explore these areas with components: {areas_with_components}.\n"
            env_obs += f"🚫 DO NOT explore area A if it's not in this list. Skip empty areas completely!\n"
        
        print(f"📝 传递给LLM的环境观察信息:\n{env_obs}")
        return env_obs  

    def replace_placeholders_with_positions(self, text: str):
        if not self.area_positions:
            return text
        def repl(m):
            label = m.group(1)
            coords = self.area_positions.get(label)
            if coords and not any(x is None for x in coords):
                return f"<{label}> at ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
            return m.group(0)
        return re.sub(r"<([A-Z])(?:\s*\(\d+\))?>", repl, text)

    def extract_area_positions_from_humanoid_file(self) -> Dict[str, Tuple[float, float, float]]:
        if not os.path.exists(self.humanoid_file_path):
            return {}
        text = open(self.humanoid_file_path, 'r', encoding='utf-8').read()
        x_matches = re.findall(r'default_pose(\d*)\.p\.x\s*=\s*([\-0-9\.]+)', text)
        y_matches = re.findall(r'default_pose(\d*)\.p\.y\s*=\s*([\-0-9\.]+)', text)
        z_matches = re.findall(r'default_pose(\d*)\.p\.z\s*=\s*([\-0-9\.]+)', text)
        pos_map = {}
        for k, v in x_matches:
            key = k if k != '' else '1'
            pos_map.setdefault(key, [None, None, None])[0] = float(v)
        for k, v in y_matches:
            key = k if k != '' else '1'
            pos_map.setdefault(key, [None, None, None])[1] = float(v)
        for k, v in z_matches:
            key = k if k != '' else '1'
            pos_map.setdefault(key, [None, None, None])[2] = float(v)
        numeric_items = sorted(((int(k), tuple(v)) for k, v in pos_map.items()), key=lambda x: x[0])
        labels = ['A', 'B', 'C', 'D']
        areas: Dict[str, Tuple[float, float, float]] = {}
        for i, item in enumerate(numeric_items[:4]):
            areas[labels[i]] = item[1]
        return areas

    def get_agent_positions_from_task(self, env_id: int = 0) -> Dict[str, Tuple[float, float]]:
        if not ISAAC_GYM_AVAILABLE or gymtorch is None:
            raise RuntimeError("isaacgym.gymtorch 未找到，请在有 Isaac Gym 环境下运行此函数")
        root_state = self.task.gym.acquire_actor_root_state_tensor(self.task.sim)
        root_state = gymtorch.wrap_tensor(root_state)
        num_actors = self.task.get_num_actors_per_env()
        agent_positions: Dict[str, Tuple[float, float]] = {}
        for name, idx in self.actor_indices_map.items():
            global_idx = env_id * num_actors + int(idx)
            pos = root_state[global_idx, :3].cpu().numpy()
            agent_positions[name] = (float(pos[0]), float(pos[1]))
        return agent_positions

    @staticmethod
    def auto_build_actor_indices_map_from_env(agent_name_to_patterns: Dict[str, Tuple[str]], actor_names: Dict[int, str]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for idx, name in actor_names.items():
            for agent_prompt, patterns in agent_name_to_patterns.items():
                if agent_prompt in result:
                    continue
                for p in patterns:
                    if p.lower() in name.lower():
                        result[agent_prompt] = idx
                        break
        return result


if __name__ == "__main__":
    print("welcome !(input 'exit' to exit)")
    while True: 
        user_input = input()
        if user_input.lower() == "exit":
            print("goodbye!")
            break