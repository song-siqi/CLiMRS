import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import numpy as np
from tqdm import tqdm
import time
import json
from openai import OpenAIError,OpenAI
import backoff
import traceback

# from llm_test.llm_module import Agent, API_KEY_R17B, API_KEY_SIQI, API_URL, API_URL_R17B, MODEL_SELECTION
from .llm_agents.oracle_planner import OraclePlanner

from types import SimpleNamespace

# @ray.remote
class ArenaMultiAgent(object):
    def __init__(
            self,
            environment_fn,
            agent_fn,
            args,
            run_predefined_actions=False
        ):
        # run_predefined_actions is a parameter that you can use predefined_actions.json to strictly set the agents' actions instead of using algorithm to calculate the action.
        
        # Used in multiple methods for environment management
        self.env_fn = environment_fn
        self.agents = agent_fn
        self.args = args
        self.num_agents = len(agent_fn)
        
        # Used in check_progress() and other goal tracking
        self.task_goal = None
        
        # Used for logging
        self.record_dir = f'./log/{args.env}.txt'
        self.debug = args.debug
        
        print("Init Env")
        # Main environment instance used throughout
        self.env = environment_fn()
        
        # Controls whether to use predefined actions from JSON
        self.run_predefined_actions = run_predefined_actions

        # Prompt paths used by oracle and agent planning
        self.oracle_prompt_path = args.oracle_prompt_path
        self.agent_selection_prompt_path = args.agent_selection_prompt_path
        self.quadrotor_prompt_path = args.quadrotor_prompt_path
        self.mobile_car_prompt_path = args.mobile_car_prompt_path
        self.humanoid_prompt_path = args.humanoid_prompt_path
        self.robot_arm_prompt_path = args.robot_arm_prompt_path

        # Dialogue tracking
        self.dialogue_history = ""
        self.total_dialogue_history = []
        
        # LLM configuration parameters
        self.chat = True
        self.source = args.source
        self.lm_id = args.lm_id
        # self.lm_id = 'gpt-3.5-turbo-1106'
        
        # Not currently used but reserved for future LLM config
        self.device = None
        self.sampling_parameters = None
        
        # Cost tracking
        self.total_cost = 0

        # State tracking for task completion
        self.last_done = False
        self.last_task_results = None
        self.last_satisfied = None
        self.last_unsatisfied = None
        self.costdict = {}

        # here is the oracle planner, change the args and check whether works properly
        self.oracle_planner = OraclePlanner(
            environment_fn=self.env_fn,
            agent_fn=self.agents,
            args=self.args,
            run_predefined_actions=self.run_predefined_actions,
            oracle_prompt_path=self.oracle_prompt_path,
            agent_selection_prompt_path=self.agent_selection_prompt_path,
            agent_grouping_prompt_path=getattr(args, 'agent_grouping_prompt_path', 'LLM/dev_revision/prompt/agent_grouping_prompt.txt'),
            agent_grouping_vanilla_prompt_path=getattr(args, 'agent_grouping_vanilla_prompt_path', 'LLM/dev_revision/prompt/agent_grouping_vanilla_prompt.txt'),
        )

    def perform_agent_grouping(self, observations, task_goal, dialogue_history):
        """
        Perform agent grouping using the oracle planner's grouping functionality.
        This integrates the Vanilla Grouping and Structured Extraction process.
        """
        try:
            grouping_result = self.oracle_planner.agent_grouping(
                observations=observations,
                task_goal=task_goal,
                dialogue_history=dialogue_history
            )
            
            if grouping_result['success']:
                print(f"✅ Agent grouping successful!")
                if self.debug:
                    print(f"Vanilla Strategy:\n{grouping_result['vanilla_strategy']}")
                    print(f"Structured Groups:\n{grouping_result['structured_groups']}")
                    
                self.total_cost += grouping_result['usage']
                return grouping_result
            else:
                print(f"❌ Agent grouping failed: {grouping_result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Agent grouping exception: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_structured_groups(self, structured_groups_output, available_agents):
        """
        Parse structured groups output into group data structures
        """
        parsed_groups = []
        
        # Create mapping from agent ID to agent info
        id_to_agent = {}
        for agent in available_agents:
            id_to_agent[agent['id']] = agent
        
        # Parse each line
        lines = structured_groups_output.strip().split('\n')
        import re
        
        for line in lines:
            if line.strip().startswith('Group ') and ':' in line and '- Sub-goal:' in line:
                try:
                    # Extract group number
                    group_match = re.match(r'Group\s+(\d+):', line)
                    if not group_match:
                        continue
                    group_id = int(group_match.group(1)) - 1  # Convert to 0-based
                    
                    # Extract agents part and sub-goal
                    parts = line.split(' - Sub-goal: ')
                    if len(parts) != 2:
                        continue
                    
                    agents_part = parts[0].split(': ', 1)[1]  # Everything after "Group X: "
                    sub_goal = parts[1].strip()
                    
                    # Extract agent IDs (filter out objects and only keep actual agents)
                    agent_pattern = r'<([^>]+)>\((\d+)\)'
                    agent_matches = re.findall(agent_pattern, agents_part)
                    
                    group_agents = []
                    valid_agent_classes = ['humanoid', 'franka', 'robot arm', 'robot_arm', 'mobile_car', 'mobile_car_1', 'mobile_car_2', 'mobile_car_3', 'quadrotor']
                    
                    for class_name, agent_id_str in agent_matches:
                        try:
                            agent_id = int(agent_id_str)
                            # Only add if it's actually an agent (not an object like wheel or trunk)
                            if agent_id in id_to_agent and any(agent_class in class_name.lower() for agent_class in ['humanoid', 'franka', 'robot', 'mobile_car', 'quadrotor']):
                                group_agents.append(id_to_agent[agent_id])
                        except ValueError:
                            continue
                    
                    if group_agents:  # Only add non-empty groups
                        parsed_groups.append({
                            'group_id': group_id,
                            'agents': group_agents,
                            'sub_goal': sub_goal
                        })
                        
                except Exception as e:
                    print(f"Error parsing group line: {line}, error: {e}")
                    continue
        
        return parsed_groups

    def _execute_group_agents(self, obs, obs2text, id_name_dict):
        """
        Execute complete group agent workflow with parallel action execution
        """
        # Extract available agents from observation
        available_agents = []
        for i in range(self.num_agents):
            agent_obs = obs[i]
            id2node = {node['id']: node for node in agent_obs["nodes"]}
            agent_id = int(self.env.id_name_dict[i][1])
            agent_node = id2node[agent_id]
            
            agent_info = {
                "agent_index": i,
                "id": agent_id,
                "class_name": agent_node["class_name"],
                "properties": agent_node["properties"],
                "states": agent_node["states"],
                "observation_text": self.agent_obs2text(obs, i)
            }
            available_agents.append(agent_info)
        
        # Perform agent grouping
        max_retries = 3
        retry_count = 0
        parsed_groups = []
        
        while retry_count < max_retries:
            try:
                # Use oracle planner to perform agent grouping
                grouping_result = self.perform_agent_grouping(
                    observations=obs2text,
                    task_goal=self.env.goal_instruction,
                    dialogue_history=self.dialogue_history
                )
                
                if grouping_result and grouping_result['success']:
                    self.total_cost += grouping_result['usage']
                    
                    # Log grouping results
                    self.write_log_to_file("Vanilla Agent Grouping Strategy: " + grouping_result['vanilla_strategy'])
                    self.write_log_to_file("Structured Agent Grouping Output: " + grouping_result['structured_groups'])
                    # Parse grouping results
                    parsed_groups = self._parse_structured_groups(grouping_result['structured_groups'], available_agents)
                    
                    if parsed_groups:  # If we have valid groups, break
                        break
                    else:
                        print(f"No valid groups found. Retry {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        
            except Exception as e:
                print(f"Failed during agent grouping (attempt {retry_count + 1}/{max_retries}): {e}")
                if "502" in str(e) or "Bad Gateway" in str(e):
                    print("Network error detected. Waiting before retry...")
                    import time
                    time.sleep(2)
                retry_count += 1

        # If all retries failed, create single group with all agents as fallback
        if retry_count >= max_retries or not parsed_groups:
            parsed_groups = [{
                'group_id': 0,
                'agents': available_agents,
                'sub_goal': "Execute the main task goal"
            }]
            self.write_log_to_file("Max retries reached. Using single group with all agents as fallback.")
            print("Max retries reached. Using single group with all agents as fallback.")

        for group in parsed_groups:
            group_agent_names = [f"<{agent['class_name']}>({agent['id']})" for agent in group['agents']]
            group_summary = f"Group {group['group_id']}: {', '.join(group_agent_names)} - Sub-goal: {group['sub_goal']}"
            self.write_log_to_file(group_summary)
            print(group_summary)

        # Perform oracle planning for each group and collect actions
        return self._execute_parallel_groups(parsed_groups, obs, id_name_dict)

    def _execute_parallel_groups(self, parsed_groups, obs, id_name_dict):
        """
        Execute oracle planning for each group and run actions in parallel
        """
        group_planning_results = []
        
        # Oracle planning for each group
        for group in parsed_groups:
            # Create observation text for this group only
            group_obs_text = ''
            for agent in group['agents']:
                group_obs_text += agent['observation_text'] + '\n'
            
            # Perform oracle planning for this group
            try:
                # Create group-specific instruction
                group_agents_list = [f"<{agent['class_name']}>({agent['id']})" for agent in group['agents']]
                group_specific_instruction = (
                    f"IMPORTANT: You are planning ONLY for Group {group['group_id']} agents: {', '.join(group_agents_list)}. "
                    f"Do NOT give instructions to agents outside this group. "
                    f"Sub-goal for this group: {group['sub_goal']}. "
                    f"Overall task: {self.env.goal_instruction}"
                )
                
                vanilla_message, usage = self.oracle_planner.oracle_planning_vanilla(
                    obs_text=group_obs_text,
                    goal_instruction=group_specific_instruction,
                    num_agents=len(group['agents']),
                    dialogue_history=self.dialogue_history,
                )
                self.total_cost += usage
                
                self.write_log_to_file(f"Vanilla Oracle Message (Group {group['group_id']}): " + vanilla_message)
                self.total_dialogue_history.append(f"Vanilla Oracle Message (Group {group['group_id']}): " + vanilla_message)
                
                # Extract structured message for this group
                message, usage = self.oracle_planner.extract_structured_message(vanilla_message)
                self.total_cost += usage
                self.write_log_to_file(f"Extracted Oracle Message (Group {group['group_id']}): " + message)
                
                # Store planning result for this group
                group_planning_results.append({
                    'group_id': group['group_id'],
                    'agents': group['agents'],
                    'sub_goal': group['sub_goal'],
                    'vanilla_message': vanilla_message,
                    'extracted_message': message
                })
                
            except Exception as e:
                print(f"Failed to get oracle planning for group {group['group_id']}: {e}")
                continue

        # Collect all actions for parallel execution
        actions_to_run = []
        agent_messages_to_log = []
        subgoals = []
        
        for group_result in group_planning_results:
            message = group_result['extracted_message']
            subgoals.append(group_result['sub_goal'])
            
            print(f"Processing group {group_result['group_id']} message: {message}")

            # Process each agent in this group
            for agent_info in group_result['agents']:
                try:
                    agent_id = agent_info['id']
                    class_name = agent_info['class_name']
                    agent_id_internal = agent_info['agent_index']
                    agent_obs = agent_info['observation_text']
                    # Determine prompt path
                    prompt_path = self._get_prompt_path(class_name)
                    
                    chat_agent_info = {
                        "class_name": class_name,
                        "id": agent_id,
                        "observation": agent_obs,
                        "instruction": message,
                        "prompt_path": prompt_path
                    }

                    try:
                        agent_action, agent_message, llm_info = self.get_actions_feedback(obs, chat_agent_info)
                        
                        # Check preconditions before adding action
                        should_execute = self._check_action_preconditions(agent_action, class_name, agent_id)
                        
                        # Only add the action if it's valid and meets preconditions
                        if agent_action is not None and should_execute:
                            actions_to_run.append({
                                "class_name": class_name,
                                "real_id": agent_id,
                                "action": agent_action,
                                "agent_id_internal": agent_id_internal,
                                "group_id": group_result['group_id']
                            })
                            agent_messages_to_log.append(agent_message)
                        elif agent_action is not None and not should_execute:
                            actions_to_run.append({
                                "class_name": class_name,
                                "real_id": agent_id,
                                "action": f"[wait] <{class_name}> ({agent_id}) wait for preconditions",
                                "agent_id_internal": agent_id_internal,
                                "group_id": group_result['group_id']
                            })
                            agent_messages_to_log.append(f"{class_name}({agent_id}): Waiting for preconditions to be met")

                        self.costdict = self.update_dict(f"<{class_name}>({agent_id})", llm_info["LLM"]["cost"], self.costdict)
                        
                        self.write_log_to_file(str(llm_info["LLM"]["action_list"]))
                        self.write_log_to_file(f"<{class_name}>({agent_id}): " + str(agent_message))
                        self.total_dialogue_history.append(f"<{class_name}>({agent_id}): " + str(agent_message))
                        
                    except Exception as agent_error:
                        print(f"Failed to get action for agent <{class_name}>({agent_id}): {agent_error}")
                        agent_message = f"<{class_name}>({agent_id}): Failed to get action due to error: {agent_error}"
                        agent_messages_to_log.append(agent_message)
                
                except Exception as e:
                    print(f"An error occurred processing agent in group {group_result['group_id']}: {e}")
                    import traceback
                    traceback.print_exc()

        self.subgoal = " | ".join(subgoals)

        for i, action in enumerate(actions_to_run):
            print(f"   {i+1}. {action['action']} (Agent: {action['class_name']}({action['real_id']}), Group: {action['group_id']})")

        # Execute actions in parallel or return for LLMManager to handle
        return self._finalize_group_execution(actions_to_run, agent_messages_to_log)

    def _check_action_preconditions(self, action, class_name, agent_id):
        """
        Check if an agent should execute the proposed action based on preconditions
        """
        if not action:
            return False
        
        # Get current agent states from environment
        if hasattr(self.env, 'task') and hasattr(self.env.task, 'agent_states'):
            agent_states = self.env.task.agent_states
        else:
            return True  # Allow execution if we can't check states
        
        # Mobile car preconditions
        if 'mobile_car' in class_name:
            agent_key = f'mobile_car_{agent_id - 200}({agent_id})'
            
            if '[move]' in action:
                # Move action: can execute if idle or any other non-conflicting state
                current_status = agent_states.get(agent_key, {}).get('status', 'idle')
                return current_status in ['idle', 'moved', 'pushed']  # Can re-move if needed
                
            elif '[push]' in action:
                # Push action: can only execute if mobile_car has moved to component location
                current_status = agent_states.get(agent_key, {}).get('status', 'idle')
                if current_status == 'moved':
                    return True
                else:
                    print(f"🚫 {agent_key} cannot push - status is '{current_status}', needs to be 'moved'")
                    return False
        
        # Franka/robot arm preconditions
        elif class_name in ['franka', 'robot arm', 'robot_arm']:
            if '[check]' in action:
                # Check action: can only execute if all mobile cars have pushed their components
                mobile_car_states = {
                    'mobile_car_1(201)': agent_states.get('mobile_car_1(201)', {}).get('status', 'idle'),
                    'mobile_car_2(202)': agent_states.get('mobile_car_2(202)', {}).get('status', 'idle'),
                    'mobile_car_3(203)': agent_states.get('mobile_car_3(203)', {}).get('status', 'idle')
                }
                
                all_pushed = all(status == 'pushed' for status in mobile_car_states.values())
                if not all_pushed:
                    print(f"🚫 Franka cannot check - mobile car states: {mobile_car_states}")
                    print(f"   Waiting for all mobile cars to reach 'pushed' status")
                    return False
                return True
                
            elif '[pick]' in action:
                # Pick action: can only execute if franka has completed check
                franka_key = f'{class_name}({agent_id})'
                current_status = agent_states.get(franka_key, {}).get('status', 'idle')
                if current_status in ['checked', 'idle']:  # Assuming 'checked' status after check completion
                    return True
                else:
                    print(f"🚫 {franka_key} cannot pick - status is '{current_status}', needs to complete check first")
                    return False
        
        # Humanoid preconditions
        elif class_name == 'humanoid':
            # Humanoid can usually act freely for obstacle clearing
            return True
        
        # Wait actions are always allowed
        if '[wait]' in action:
            return True
            
        # Default: allow other actions
        return True

    def _get_prompt_path(self, class_name):
        """Get appropriate prompt path for agent class"""
        if class_name == 'quadrotor':
            return self.quadrotor_prompt_path
        elif class_name in ['mobile_car', 'mobile_car_1', 'mobile_car_2', 'mobile_car_3']:
            return self.mobile_car_prompt_path
        elif class_name == 'humanoid':
            return self.humanoid_prompt_path
        elif class_name in ['robot arm', 'robot_arm', 'franka']:
            return self.robot_arm_prompt_path
        else:
            return self.mobile_car_prompt_path  # fallback

    def _finalize_group_execution(self, actions_to_run, agent_messages_to_log):
        """Finalize the group execution and return results"""
        if not actions_to_run:
            done = self.last_done
            task_results = self.last_task_results
            satisfied = self.last_satisfied
            unsatisfied = self.last_unsatisfied
            self.env.steps += 1
            id, agent_action, agent_message = [], None, "all robot agents: Group execution failed - no valid actions generated."
            self.total_dialogue_history.append(agent_message)
        else:
            # Store all actions for parallel execution by LLMManager
            if hasattr(self.env, 'parallel_actions'):
                self.env.parallel_actions = actions_to_run
                print(f"📦 Stored {len(actions_to_run)} actions in parallel_actions")
            else:
                print("⚠️ env.parallel_actions attribute not found! Cannot store parallel actions.")
            
            # Execute the first action as the main action for compatibility
            main_action = actions_to_run[0]
            try:
                done, task_results, satisfied, unsatisfied, steps = self.env.step(
                    main_action['class_name'], 
                    main_action['real_id'], 
                    main_action['action'], 
                    self.task_goal
                )
                self.last_done = done
                self.last_task_results = task_results
                self.last_satisfied = satisfied
                self.last_unsatisfied = unsatisfied
                
                # Return info from the main action
                id = [main_action['agent_id_internal']]
                agent_action = main_action['action']
                agent_message = agent_messages_to_log[0] if agent_messages_to_log else "Group execution completed"
                
                # Log all parallel actions for debugging
                print(f"🚀 Main action executed: {main_action['action']}")
                if len(actions_to_run) > 1:
                    print(f"📦 Stored {len(actions_to_run)-1} parallel actions for execution:")
                    for i, action in enumerate(actions_to_run[1:], 1):
                        print(f"   {i}. {action['action']} (Group {action['group_id']})")
                
            except Exception as e:
                print(f"Exception occurs when performing main action: {main_action['action']}")
                raise Exception

        # Log costs and dialogue history
        self.write_log_to_file(f"COST1:{self.total_cost}!!!!!")
        self.write_log_to_file(str(self.costdict))
        self.write_log_to_file(f"COST2:{sum(self.costdict.values())}!!!!!")
        self.write_log_to_file(f'总的花费：{self.total_cost + sum(self.costdict.values())}')
        self.write_log_to_file('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ')
        
        numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
        self.dialogue_history = '\n'.join(numbered_list[-10:])
        self.write_log_to_file(f'\nDIALOGUE_HISTORY:\n{self.dialogue_history}')  
        
        steps = self.env.steps
        return done, task_results, satisfied, unsatisfied, id, agent_action, agent_message, steps

    def get_actions_feedback(self, obs, chat_agent_info):

        for id, agent in enumerate(self.agents):
            if agent.agent_node["id"] == chat_agent_info["id"]:

                action, message, info = agent.get_action(obs[id], chat_agent_info, self.env.task_goal)

        return action, message, info

    def agent_obs2text(self, observation, agent_id):
        '''
        Add the observation of the agent to the text, which is based on hardcoding translation
        Return:
            text: str, the text of the agent's observation
        '''
        text = ""
        observation = observation[agent_id]
       
        id2node = {node['id']: node for node in observation["nodes"]}
        agent_class = id2node[int(self.env.id_name_dict[agent_id][1])]["class_name"]
        with_quadrotor_id = None
        for node in observation["nodes"]:
            if node["category"] == "Agents" and self.env.id_name_dict[agent_id][1] == node["id"]:
                # agent_node = node
                text += "I am <" + node["class_name"] +">(" + str(node["id"]) + "). "   
                if len(node['states']) != 0:
                    states = ', '.join(node['states'])
                    text += "Now my state is: " + states + ". "
                for edge in observation["edges"]:
                    if edge["from_id"] == node["id"]:
                        text += "I am " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). "
                    if edge["relation_type"] == "WITH":
                        with_quadrotor_id = edge["to_id"]
                text += '\n'


        for node in observation["nodes"]:
            if node["category"] == "Rooms" and node["id"] == observation["agent_in_room_id"]:
                text += "Now I am in the <"+ node["class_name"] +">(" + str(node["id"]) + "). In this room, I can see : \n"
        for node in observation["nodes"]:
            if node["id"] != self.env.id_name_dict[agent_id][1] and node["category"] != "Rooms":
                text += "<" + node["class_name"] +">(" + str(node["id"]) + "). "
                if len(node['properties']) != 0:
                    properties = ', '.join(node['properties'])
                    text += "Its properties are: " + properties + ". "
                if len(node['states']) !=0 :
                    states = ', '.join(node['states'])
                    text += "Now its state is: " + states + ". \n"
                else:
                    text += '\n'
        text += "These objects have a certain position relationship with each other: \n"
        for node in observation["nodes"]:
            if node["id"] != self.env.id_name_dict[agent_id][1] and node["category"] != "Rooms":
                for edge in observation["edges"]:
                    if edge["from_id"] == node["id"]: 
                    # if edge["from_id"] == node["id"] and edge["relation_type"] != "WITH" : #WITH is exclusive to quadrotor and basket
                        if edge["from_id"] == with_quadrotor_id and agent_class == 'quadrotor':
                            text += "The <" + node["class_name"] +">(" + str(node["id"]) + ") is with me LAND " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        elif edge["relation_type"] == "LEADING TO":
                            text += "The <" + node["class_name"] +">(" + str(node["id"]) + ") is " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        else:
                            text += "The <" + node["class_name"] +">(" + str(node["id"]) + ") is " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
        for edge in observation["edges"]:
            if edge["relation_type"] == "WITH" and agent_class == 'quadrotor':
                in_basket = False
                text += "I have a <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + ") with me. " 
                for edges in observation["edges"]:
                    if edges["to_id"] == edge["to_id"] and edges["relation_type"] == "INSIDE" :
                        text += "<" + id2node[edges["from_id"]]["class_name"] + ">("+ str(edges["from_id"]) + ") is in my <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        in_basket = True    
                if in_basket == False:
                    text += "But nothing is in my <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
            if edge["relation_type"] == "HOLD" and agent_class != 'quadrotor':
                text += "I am holding a <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + ") in my hand. \n"    
        
        # Add agent status information for Oracle decision making
        if hasattr(self.env, 'agent_states') and self.env.agent_states:
            text += "\nAgent Status Information:\n"
            for agent_key, agent_state in self.env.agent_states.items():
                text += f"{agent_key}: {agent_state}\n"
        
        # print(text)
        return text
    
    def write_log_to_file(self,log_message, file_name = None):
        file_name = self.record_dir
        with open(file_name, 'a', encoding='utf-8') as file:  
            file.write(log_message + '\n')  

    def step(self):
        '''
        Perform one step of the arena multi-agent system.
        Returns:
            done: bool, whether the task is done
            task_results: list, the results of the task
            satisfied: bool, whether the task is satisfied
            unsatisfied: bool, whether the task is unsatisfied
            id: list, the id of the agent
        '''
        if self.env.steps == 0:
            pass

        obs = self.env.get_observations()
        id_name_dict = self.env.id_name_dict
        # NOTE: translate the observation of the agent to text as part of the oracle prompt
        obs2text = ''
        for i in range(self.num_agents):
            obs2text += self.agent_obs2text(obs, i) + '\n'

        # here starts this step in the loop
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print((f"@@@@@@@@@@@@@@@@@@@@@@@@ Task_ID: {self.env.task_id} @@@@@@@@@@@"))
        print(f"$$$$$$$$$$$$$$$$$$$$$$$ Step:{self.env.steps} $$$$$$$$$$$$$$$$$$$$$$$")
        print(self.env.goal_instruction)

        # here logs the head of this loop in the log file
        self.write_log_to_file(f"@@@@@@@@@@@@@@@@@@@@@@@ Task_ID: {self.env.task_id} @@@@@@@@@@@")
        self.write_log_to_file(f"$$$$$$$$$$$$$$$$$$$$$$$ Step:{self.env.steps} $$$$$$$$$$$$$$$$$$$$$$$")
        self.write_log_to_file(f'''*******************************************************************************************
                               TASK_GOAL: {self.env.goal_instruction}
                               ''')
        self.write_log_to_file("FULL OBSERVATIONS: \n" + obs2text)

        if getattr(self.args, 'group_agents', False):
            '''
            Agent grouping process (complete group execution):
            1. Extract available agents from observation
            2. Perform agent grouping using oracle planner
            3. Parse grouping results into separate groups
            4. For each group, perform oracle planning separately
            5. Execute all group actions in parallel
            '''
            return self._execute_group_agents(obs, obs2text, id_name_dict)

        elif self.args.select_agents:
            '''
            Agent selection process:
            1. Extract available agents from observation
            2. Create agent selection prompt and use oracle planner to select agents
            3. Parse selected agent indices
            4. Create filtered observation text for selected agents only
            5. Perform oracle planning for selected agents
            '''
            # Extract available agents from observation
            available_agents = []
            for i in range(self.num_agents):
                agent_obs = obs[i]
                id2node = {node['id']: node for node in agent_obs["nodes"]}
                agent_id = int(self.env.id_name_dict[i][1])
                agent_node = id2node[agent_id]
                
                agent_info = {
                    "agent_index": i,
                    "id": agent_id,
                    "class_name": agent_node["class_name"],
                    "properties": agent_node["properties"],
                    "states": agent_node["states"],
                    "observation_text": self.agent_obs2text(obs, i)
                }
                available_agents.append(agent_info)
            
            # Create agent selection prompt
            agents_info_text = ""
            for agent in available_agents:
                agents_info_text += f"<{agent['class_name']}>({agent['id']}) - Properties: {agent['properties']}, States: {agent['states']}\n"
            
            # print(f"agents_info_text: {agents_info_text}")
            # for example (agents_info_text):
            # <quadrotor>(22) - Properties: ['MOVABLE', 'FLYABLE', 'HAVE_A_BASKET'], States: ['LAND']
            # <mobile_car>(202) - Properties: ['MOVABLE'], States: []
            # <humanoid>(101) - Properties: ['MOVABLE'], States: []
            # <robot_arm>(606) - Properties: ['ON_HIGH_SURFACE'], States: []
            
            # NOTE: here is the process of the agent selection
            max_retries = 3
            retry_count = 0
            selected_indices = []
            
            while retry_count < max_retries:
                # Use oracle planner to select agents
                selection_output, usage = self.oracle_planner.agent_selection(
                    agents_info_text=agents_info_text,
                    goal_instruction=self.env.goal_instruction,
                    obs_text=obs2text,
                    dialogue_history=self.dialogue_history,
                )
                self.total_cost += usage
                
                self.write_log_to_file("Agent Selection Output: " + selection_output)
                print(f"selection_output: {selection_output}")

                # Parse selected agent IDs and convert to indices
                try:
                    selected_agent_ids = [int(x.strip()) for x in selection_output.split(",")]
                    
                    # Create mapping from agent ID to index
                    id_to_index = {}
                    for i in range(self.num_agents):
                        agent_id = int(self.env.id_name_dict[i][1])
                        id_to_index[agent_id] = i
                    
                    # Convert selected IDs to indices
                    selected_indices = []
                    for agent_id in selected_agent_ids:
                        if agent_id in id_to_index:
                            selected_indices.append(id_to_index[agent_id])
                        else:
                            print(f"Warning: Agent ID {agent_id} not found in available agents")
                    
                    if selected_indices:  # If we have valid selections, break
                        break
                    else:
                        print(f"No valid agent IDs found. Retry {retry_count + 1}/{max_retries}")
                        retry_count += 1
                except:
                    print(f"Failed to parse agent selection. Retry {retry_count + 1}/{max_retries}")
                    retry_count += 1
            
            # If all retries failed, use all agents as fallback
            if retry_count >= max_retries:
                selected_indices = list(range(self.num_agents))
                self.write_log_to_file("Max retries reached. Using all agents as fallback.")
                print("Max retries reached. Using all agents as fallback.")

            # Log both selected IDs and indices for clarity
            selected_ids_for_log = [int(self.env.id_name_dict[i][1]) for i in selected_indices]
            self.write_log_to_file(f"Selected Agent IDs: {selected_ids_for_log}")
            # self.write_log_to_file(f"Selected Agent Indices: {selected_indices}")
            print(f"Selected Agent IDs: {selected_ids_for_log}")
            # print(f"Selected Agent Indices: {selected_indices}")
            
            # import ipdb; ipdb.set_trace()

            # Create filtered observation text for selected agents only
            selected_obs_text = ''
            for i in selected_indices:
                selected_obs_text += self.agent_obs2text(obs, i) + '\n'
            
            # print(f"selected_obs_text:\n{selected_obs_text}")
            # import ipdb; ipdb.set_trace()

            # Perform oracle planning for selected agents
            vanilla_message, usage = self.oracle_planner.oracle_planning_vanilla(
                obs_text=selected_obs_text,
                goal_instruction=self.env.goal_instruction,
                num_agents=len(selected_indices),
                dialogue_history=self.dialogue_history,
            )
            self.total_cost += usage
            self.write_log_to_file("Vanilla Oracle Message (Selected Agents): " + vanilla_message)
            self.total_dialogue_history.append("Vanilla Oracle Message (Selected Agents): " + vanilla_message)
            
            message, usage = self.oracle_planner.extract_structured_message(vanilla_message)
            self.total_cost += usage
            self.write_log_to_file("Extracted Oracle Message (Selected Agents): " + message)
            self.subgoal = message

        else:   
            # NOTE: here organizes the oracle prompt and the structured message extraction for the oracle agent
            vanilla_message, usage = self.oracle_planner.oracle_planning_vanilla(
                obs_text=obs2text,
                goal_instruction=self.env.goal_instruction,
                num_agents=self.env.num_agent,
                dialogue_history=self.dialogue_history,
            )
            self.total_cost += usage
            self.write_log_to_file("Vanilla Oracle Message: " + vanilla_message)
            self.total_dialogue_history.append( "Vanilla Oracle Message: " + vanilla_message)
            
            message, usage = self.oracle_planner.extract_structured_message(vanilla_message)
            self.total_cost += usage
            self.write_log_to_file("Extracted Oracle Message: " + message)
            self.subgoal = message
            # For example: '
            # Hello <robot arm>(24): Please pick up the <bread>(26) from the <microwave>(15) and place it into the <plate>(51). 
            # Then, open the <microwave>(15), and place the <milkbox>(30) inside it.'

        # print(f"message: {message}")
        # import ipdb; ipdb.set_trace()

        # debug with the oracle prompt and the extracted message
        if self.debug:
            oracle_prompt = self.oracle_planner.get_oracle_prompt(
                obs_text=obs2text,
                goal_instruction=self.env.goal_instruction,
                num_agents=self.env.num_agent,
                dialogue_history=self.dialogue_history,
            )
            # input('wait a minute!')
            print(f"message_oracle_prompt:\n{oracle_prompt}")
            print('\n')
            print(f"message_oracle_outputs:\n{message}")
        
        # NOTE: here is the process of the single feedback agent doing the action extraction and feedback
        try:
            start_class_name = message.find('<') + 1
            end_class_name = message.find('>')
            start_id = message.find('(') + 1
            end_id = message.find(')')

            # extract class_name and real_id
            class_name = message[start_class_name:end_class_name]
            try:
                real_id = int(message[start_id:end_id])
            except ValueError as e:
                return False, [], [], [], 0, "None", f"Parse error: {e}", 0
            
            id  = [key for key, value in id_name_dict.items() if value[1] == real_id]
            if not id:
                return False, [], [], [], 0, "None", "Agent ID not found", 0
            agent_obs = self.agent_obs2text(obs, id[0])

            if class_name == 'quadrotor':
                prompt_path = self.quadrotor_prompt_path
            elif class_name == 'mobile_car':
                prompt_path = self.mobile_car_prompt_path
            elif class_name == 'humanoid':
                prompt_path = self.humanoid_prompt_path
            elif class_name == 'robot arm' or class_name == 'robot_arm':
                prompt_path = self.robot_arm_prompt_path
            else:
                # Default fallback to mobile_car
                prompt_path = self.mobile_car_prompt_path

            chat_agent_info = {
                "class_name": class_name, 
                "id": real_id, 
                "observation": agent_obs, 
                "instruction": message, 
                "prompt_path": prompt_path
            }

            agent_action, agent_message, agent_info = self.get_actions_feedback(obs, chat_agent_info)
            # agent_action = '[movetowards] <coffeetable> (12)'
            # agent_message = 'YES I CAN. \n\nThe first step to accomplish the task is to move towards the coffeetable. This action is available in the list of actions I can perform. After moving to the coffeetable, I can use my robotic arm to pick up the apple. \n\nTherefore, the best available action to achieve the goal as soon as possible is to [movetowards] <coffeetable> (12). The action I finally decided to perform is [movetowards] <coffeetable> (12).'
            
            self.costdict =  self.update_dict(f"<{class_name}>({real_id})", agent_info["LLM"]["cost"], self.costdict)

            self.write_log_to_file(str(agent_info["LLM"]["action_list"]))
            self.write_log_to_file(f"<{class_name}>({real_id}): " + str(agent_message))
            self.write_log_to_file(f"COST1:{self.total_cost}!!!!!")
            self.write_log_to_file(str(self.costdict))
            self.write_log_to_file(f"COST2:{sum(self.costdict.values())}!!!!!")
            self.write_log_to_file(f'总的花费：{self.total_cost + sum(self.costdict.values())}')
            self.write_log_to_file('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ')

            self.total_dialogue_history.append(f"<{class_name}>({real_id}): " + str(agent_message))
            numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
            self.dialogue_history = '\n'.join(numbered_list[-10:])
        except Exception as e:
    
            print(f"An error occurred: {e}")
            traceback.print_exc()
            error_info = traceback.format_exc()
            self.write_log_to_file(f"An error occurred: {e}")
            self.write_log_to_file(error_info+'\n\n')
            agent_action = None
            agent_message = "all robot agents: In the last step, the oracle's reasoning was incorrect, and no instructions were given to any of the robot agents, therefore none of the robot agents performed any actions. Please reassess the information in the environment and give a correct instruction strictly following the template 'Hello <class name>(id): #message#.'"
            self.total_dialogue_history.append(agent_message)
            numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
            self.dialogue_history = '\n'.join(numbered_list[-10:])
      
        # NOTE: here is the process of the environment interactions
        if agent_action is None:
            done = self.last_done
            task_results = self.last_task_results
            satisfied = self.last_satisfied
            unsatisfied = self.last_unsatisfied
            self.env.steps += 1
        else:
            try:
                done, task_results, satisfied, unsatisfied, steps = self.env.step(class_name, real_id, agent_action, self.task_goal)
                self.last_done = done
                self.last_task_results = task_results
                self.last_satisfied = satisfied
                self.last_unsatisfied = unsatisfied
                
            except Exception as e:
                print("Exception occurs when performing action: ", agent_action)
                raise Exception
        self.write_log_to_file(f'\nDIALOGUE_HISTORY:\n{self.dialogue_history}')  
        steps = self.env.steps
        agent_id = id[0] if 'id' in locals() and id else 0
        return done, task_results, satisfied, unsatisfied, agent_id, agent_action, agent_message, steps


    def run(self):
        '''
        Run the arena multi-agent system.
        Returns:
            success: bool, whether the task is successful
            steps: int, the number of steps taken
            saved_info: list, a list of dictionaries containing information about each step
        '''
        self.task_goal = copy.deepcopy(self.env.task_goal)
        saved_info = []

        success = False
        while True:
            # NOTE: this is the main loop of the multi-agent system in which the plans are executed step by step
            done, task_results, satisfied, unsatisfied, id, agent_action, agent_message,steps  = self.step()
            saved_info.append({
                'task_id': self.env.task_id,
                'env_id': self.env.env_id,
                'task_name': self.env.task_name,
                'gt_steps': self.env.ground_truth_step_num,
                'task_goal': self.task_goal,
                'goal_instruction': self.env.goal_instruction,
                'step': steps,
                'subgoal': self.subgoal,
                'agent_id': id[0],
                'action': agent_action,
                'agent_message': agent_message,
                'satisfied': satisfied,
                'unsatisfied': unsatisfied,
                'env_graph': self.env.graph, 
            })
            success = done

            # NOTE: to decide whether time is up or succeed to exit the loop
            max_step = 2 * self.env.ground_truth_step_num
            if self.env.steps > max_step:
                print("---------------------------")
                print("The task failed, exceeding 2 times the number of GT steps")
                print(f"Whether steps in gt*2+1 are successful:{done}")
                print(f" setps: {steps}")
                print("---------------------------")
                self.write_log_to_file(f'''---------------------------
                                       The task failed, exceeding 2 times the number of GT steps
                                       Whether steps in gt*2+1 are successful:{done}
                                       setps: {steps}
                                       ---------------------------
                                       ''')
            
                success = False
                break
            
            if success:
                self.write_log_to_file(f'''-------------------------------------
                                            success!
                                            setps: {steps}
                                            --------------------------------
                                            ''')
                break
        saved_info[steps-1]['is_finished'] = success
        
        return success, steps, saved_info

    def update_dict(self,key, value, my_dict):

        my_dict[key] = value
        return my_dict