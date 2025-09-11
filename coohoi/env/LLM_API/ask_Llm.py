import requests
import json
import re
import os
from typing import Dict, Tuple
from isaacgym import gymtorch

class LLMWorkflow:
    def __init__(self, area_positions: Dict[str, Tuple[float, float, float]], 
                 agent_positions: Dict[str, Tuple[float, float]], 
                 actor_indices_map: Dict[str, int]):
        self.area_positions = area_positions
        self.agent_positions = agent_positions
        self.actor_indices_map = actor_indices_map
        self.API_URL = "http://35.220.164.252:3888/v1/chat/completions"  
        self.API_KEY = "sk-VS6OzWyx7SyeeNnWRo7BuUeD9H9jzxU88z9IQlcf4K72l14U"
        
    def ask_llm(self, question):
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.API_KEY}',
            'User-Agent':'Apifox/1.0.0(https://apifox.com)',
            'Content-Type': 'application/json'
        }
        # instruction head + goal description + state description + action list
        prompt = """{}
                           You are an expert in robotics, now we urgently need to use wheeled robots, 
                           humanoid robots and mechanical arm (franka) to jointly complete a small robot assembly task. 
                           Please help me design the fastest workflow to complete this task based on the skills provided by the following robots, and every step you should give the Skill list number, like A., S.. 
                           Please note that all skills and objects are represented by <name> (id), for example, <humanoid> (101).
                           Goal Description: Locate all the parts that need to be assembled (a total of 3), including 1 <trunk> (303) and 2 wheels <left wheel> (405), <right wheel> (406), 
                           and move the parts to the side of the <franka> (606) to achieve sequential assembly.
                           
                           WORKFLOW LOGIC:
                           1. First: Use A/B/C/D skills to explore different areas and find components
                           2. Second: Use E/F/G skills to move robots to franka area for component delivery
                           3. Third: Use J/K/L skills to push components to franka for assembly
                    State Description:
                    component list: <trunk> (303), <left wheel> (405), <right wheel> (406), <obstacles> (507).
                    agent list: <wheeled robot1> (202),<wheeled robot2> (203),<wheeled robot3> (204), <humanoid> (101), <franka> (606).
                    franka position: <franka> (606) is fixed in (0.0, -2.0).
                    Current Environment Observation:
                    {}
                    Skill list:
                    A. [observe] area <A> (001) - only observe, robot doesn't move
                    B. [observe] area <B> (002) - only observe, robot doesn't move  
                    C. [observe] area <C> (003) - only observe, robot doesn't move
                    D. [observe] area <D> (004) - only observe, robot doesn't move
                    E. [move] <wheeled robot1> (202) move to component location using RRT path
                    F. [move] <wheeled robot2> (203) move to component location using RRT path
                    G. [move] <wheeled robot3> (204) move to component location using RRT path
                    H. [walk] <humanoid> (101) move to selected area
                    I. [carry] <humanoid> (101) carry <obstacles> (507)
                    J. [push] <wheeled robot1> (202) push selected component to franka area
                    K. [push] <wheeled robot2> (203) push selected component to franka area
                    L. [push] <wheeled robot3> (204) push selected component to franka area
                    M. [check] <franka> (606) check <trunk> (303)
                    N. [check] <franka> (606) check <left wheel> (405)
                    O. [check] <franka> (606) check <right wheel> (406)
                    P. [pick] <franka> (606) pick and place <left wheel> (405) on <trunk> (303)
                    Q. [pick] <franka> (606) pick and place <right wheel> (406) on <trunk> (303)
                    R. [wait] <franka> (606) wait
                    S. [wait] <humanoid> (101) wait
                    T. [wait] <wheeled robot1> (202) wait
                    U. [wait] <wheeled robot2> (203) wait
                    V. [wait] <wheeled robot3> (204) wait
                    Answer: Let's think step by step.
                """

        formatted_question = self.build_prompt_with_positions(question)
        prompt_filled = prompt.format(question, formatted_question)
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
        if gymtorch is None:
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