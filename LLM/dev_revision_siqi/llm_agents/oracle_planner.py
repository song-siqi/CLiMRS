import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import copy
import numpy as np
from tqdm import tqdm
import time
import json
from openai import OpenAIError,OpenAI
import backoff
import traceback

from llm_utils.llm_module import Agent, API_KEY_R17B, API_KEY_SIQI, API_URL, API_URL_R17B, MODEL_SELECTION

from types import SimpleNamespace

class OraclePlanner(object):
    def __init__(
            self,
            environment_fn,
            agent_fn,
            args,
            run_predefined_actions=False,
            oracle_prompt_path=None,
            agent_selection_prompt_path=None,
            agent_grouping_prompt_path=None,
        ):
        self.env_fn = environment_fn
        self.agents = agent_fn
        self.args = args
        self.num_agents = len(agent_fn)
        self.task_goal = None

        # define some args for the oracle planner
        self.oracle_prompt_path = oracle_prompt_path
        self.agent_selection_prompt_path = agent_selection_prompt_path
        self.agent_grouping_prompt_path = agent_grouping_prompt_path
        # Add vanilla grouping prompt path
        if agent_grouping_prompt_path:
            self.agent_grouping_vanilla_prompt_path = agent_grouping_prompt_path.replace('agent_grouping_prompt.txt', 'agent_grouping_vanilla_prompt.txt')
        else:
            self.agent_grouping_vanilla_prompt_path = None

        # define parameters that were used but not initialized
        self.source = args.source
        self.lm_id = args.lm_id
        self.device = None
        self.chat = True
        self.debug = args.debug

        # define the llm engine and generator for the oracle agent
        if self.source == 'openai':
            api_key = args.api_key # your openai api key
            organization=args.organization # your openai organization

            client = OpenAI(api_key = api_key, organization=organization)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": args.top_p,
                    "n": args.n
                }
        elif self.source == 'llm_module':

            api_key = API_KEY_SIQI
            api_url = API_URL
            model = MODEL_SELECTION
            # model = "gpt-4o-2024-11-20"

            client = Agent(model=model, api_url=api_url, api_key=api_key)
            if self.chat:
                self.sampling_params = {
                    "max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": 1.0,
                    "n": args.n
                }
            # self.device = args.device
            # self.lm_id = args.lm_id
            # self.chat = True
            # self.llm_module = Agent(model=self.lm_id, device=self.device)
            # self.sampling_params['model'] = self.lm_id

        def lm_engine(source, lm_id, device):

            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            prompt.insert(0,{"role":"system", "content":"You are a helper assistant."})
                            response = client.chat.completions.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            if self.debug:
                                with open(f"./chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response.choices[i].message.content for i in
                                                    range(sampling_params['n'])]
                            if 'gpt-4-0125-preview' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.01 / 1000 + response.usage.completion_tokens * 0.03 / 1000
                            elif 'gpt-3.5-turbo-1106' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == 'llm_module':
                    try:
                        if self.chat:
                            prompt.insert(0,{"role":"system", "content":"You are a helper assistant."})
                            response = client.respond_once_all_args(
                                messages=prompt, **sampling_params
                            )
                            # response = SimpleNamespace(**response)
                            if self.debug:
                                with open(f"./chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['message']['content'] for i in range(sampling_params['n'])]
                            if 'gpt-4-0125-preview' in self.lm_id or 'gpt-4o-2024-11-20' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.01 / 1000 + response['usage']['completion_tokens'] * 0.03 / 1000
                            elif 'gpt-3.5-turbo-1106' in self.lm_id:
                                usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except Exception as e:
                        print(e)
                        raise e

                else:
                    raise ValueError("invalid source")
                return generated_samples, usage 
            
            return _generate

        self.generator = lm_engine(self.source, self.lm_id, self.device)

    def get_agent_grouping_vanilla_prompt(self, agents_info_text, goal_instruction, obs_text, dialogue_history):
        '''
        Get the agent grouping vanilla prompt for first-stage comprehensive planning
        '''
        with open(self.agent_grouping_vanilla_prompt_path, 'r') as f:
            agent_grouping_vanilla_prompt = f.read()
        agent_grouping_vanilla_prompt = agent_grouping_vanilla_prompt.replace('#AGENTS_INFO#', agents_info_text)
        agent_grouping_vanilla_prompt = agent_grouping_vanilla_prompt.replace('#TASK_GOAL#', goal_instruction)
        agent_grouping_vanilla_prompt = agent_grouping_vanilla_prompt.replace('#OBSERVATIONS#', obs_text)
        agent_grouping_vanilla_prompt = agent_grouping_vanilla_prompt.replace('#DIALOGUE_HISTORY#', dialogue_history)

        return agent_grouping_vanilla_prompt

    def get_agent_grouping_prompt(self, vanilla_grouping_output, agents_info_text):
        '''
        Get the agent grouping prompt for second-stage structured extraction
        '''
        with open(self.agent_grouping_prompt_path, 'r') as f:
            agent_grouping_prompt = f.read()
        agent_grouping_prompt = agent_grouping_prompt.replace('#VANILLA_GROUPING_OUTPUT#', vanilla_grouping_output)
        agent_grouping_prompt = agent_grouping_prompt.replace('#AGENTS_INFO#', agents_info_text)

        return agent_grouping_prompt
    
    def agent_grouping_vanilla(
            self,
            agents_info_text,
            goal_instruction,
            obs_text,
            dialogue_history,
        ):
        '''
        Perform vanilla agent grouping (first stage) using LLM to develop comprehensive grouping strategy.
        Args:
            agents_info_text: str, formatted text containing information about available agents
            goal_instruction: str, the goal instruction of the task
            obs_text: str, the text of the observation of the environment
            dialogue_history: str, the dialogue history
        Returns:
            vanilla_grouping_output: str, the detailed grouping strategy from LLM
            usage: float, the cost of the llm engine
        '''
        # Get the agent grouping vanilla prompt
        agent_grouping_vanilla_prompt = self.get_agent_grouping_vanilla_prompt(agents_info_text, goal_instruction, obs_text, dialogue_history)
        
        chat_prompt = [{"role": "user", "content": agent_grouping_vanilla_prompt}]
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        vanilla_grouping_output = outputs[0]

        return vanilla_grouping_output, usage

    def extract_structured_grouping(self, vanilla_grouping_output, agents_info_text):
        '''
        Extract structured grouping from vanilla grouping output (second stage).
        Args:
            vanilla_grouping_output: str, the detailed grouping strategy from first stage
            agents_info_text: str, formatted text containing information about available agents
        Returns:
            structured_grouping_output: str, the structured grouping output following exact format
            usage: float, the cost of the llm engine
        '''
        # Get the structured grouping prompt
        agent_grouping_prompt = self.get_agent_grouping_prompt(vanilla_grouping_output, agents_info_text)
        
        chat_prompt = [{"role": "user", "content": agent_grouping_prompt}]
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        structured_grouping_output = outputs[0]

        return structured_grouping_output, usage

    def agent_grouping(
            self,
            agents_info_text,
            goal_instruction,
            obs_text,
            dialogue_history,
        ):
        '''
        Perform two-stage agent grouping using LLM to divide agents into non-interfering groups.
        Args:
            agents_info_text: str, formatted text containing information about available agents
            goal_instruction: str, the goal instruction of the task
            obs_text: str, the text of the observation of the environment
            dialogue_history: str, the dialogue history
        Returns:
            vanilla_grouping_output: str, the detailed grouping strategy from LLM
            structured_grouping_output: str, the structured grouping output following exact format
            total_usage: float, the total cost of both LLM calls
        '''
        # Stage 1: Vanilla grouping for comprehensive strategy
        vanilla_grouping_output, vanilla_usage = self.agent_grouping_vanilla(
            agents_info_text, goal_instruction, obs_text, dialogue_history
        )
        
        # Stage 2: Extract structured grouping
        structured_grouping_output, structured_usage = self.extract_structured_grouping(
            vanilla_grouping_output, agents_info_text
        )
        
        total_usage = vanilla_usage + structured_usage
        
        return vanilla_grouping_output, structured_grouping_output, total_usage

    def parse_grouping_result(self, grouping_output, available_agents):
        '''
        Parse the LLM grouping output into structured groups.
        Args:
            grouping_output: str, the raw output from LLM containing agent groups and sub-goals
            available_agents: list, list of available agent dictionaries with agent info
        Returns:
            parsed_groups: list, list of dictionaries containing group information
                Format: [
                    {
                        'group_id': int,
                        'agents': [{'agent_index': int, 'id': int, 'class_name': str, ...}, ...],
                        'sub_goal': str
                    },
                    ...
                ]
        '''
        parsed_groups = []
        
        # Create mapping from agent ID to agent info
        id_to_agent = {}
        for agent in available_agents:
            id_to_agent[agent['id']] = agent
        
        # Define agent pattern at the start
        agent_pattern = r'<([^>]+)>\((\d+)\)'
        
        # Track which agents have been assigned to prevent duplicates
        assigned_agents = set()
        
        # Split text into sections by newlines and filter out empty lines
        sections = [line.strip() for line in grouping_output.split('\n') if line.strip()]
        
        group_id = 0
        
        # Main pattern to match "Group X: agent1, agent2 - Sub-goal: description"
        group_pattern = r'Group\s+\d+:\s*((?:<[^>]+>\(\d+\)(?:,\s*)?)+)(?:\s*-\s*Sub-goal:\s*([^-\n]+))?'
        
        for section in sections:
            if not section or 'non-assigned' in section.lower():
                continue
                
            group_match = re.match(group_pattern, section, re.IGNORECASE)
            if group_match:
                agents_part = group_match.group(1)
                sub_goal = group_match.group(2)
                
                # Clean up sub-goal
                if not sub_goal:
                    sub_goal = "Execute assigned tasks"
                else:
                    sub_goal = sub_goal.strip()
                    if sub_goal.endswith('.'):
                        sub_goal = sub_goal[:-1]
                
                # Parse agents
                group_agents = []
                agent_matches = re.findall(agent_pattern, agents_part)
                
                for class_name, agent_id_str in agent_matches:
                    try:
                        agent_id = int(agent_id_str)
                        # Skip if agent already assigned or not in available agents
                        if agent_id in assigned_agents or agent_id not in id_to_agent:
                            continue
                        group_agents.append(id_to_agent[agent_id])
                        assigned_agents.add(agent_id)
                    except ValueError:
                        continue
                
                if group_agents:  # Only add non-empty groups
                    parsed_groups.append({
                        'group_id': group_id,
                        'agents': group_agents,
                        'sub_goal': sub_goal
                    })
                    group_id += 1
        
        # We don't need to create a group with remaining unassigned agents, 
        # except when the grouping output is None, which means no groups were parsed successfully
        unassigned_agents = [
            agent for agent in available_agents 
            if agent['id'] not in assigned_agents
        ]
        
        if parsed_groups is None and unassigned_agents:
            parsed_groups.append({
                'group_id': group_id,
                'agents': unassigned_agents,
                'sub_goal': "Execute remaining task goals"
            })
        
        return parsed_groups

    def get_oracle_prompt(self, obs_text, goal_instruction, num_agents, dialogue_history, group_info=None):
        '''
        Get the oracle prompt
        Args:
            obs_text: str, the text of the observation of the agents
            goal_instruction: str, the goal instruction of the task
            num_agents: int, the number of agents
            dialogue_history: str, the dialogue history (deprecated)
            group_info: dict, optional information about the group being planned for
        '''
        with open(self.oracle_prompt_path, 'r') as f:
            oracle_prompt = f.read()
        
        # If group info is provided, modify the prompt to focus on group agents
        if group_info:
            # Create agent capabilities text based on group agents
            agent_capabilities = []
            agent_rules = []
            for agent in group_info['agents']:
                if agent['class_name'] == 'quadrotor':
                    agent_capabilities.append(
                        f"quadrotor: The quadrotor can take off, land, and fly in the air. The quadrotor has a basket that can be used to transport objects. "
                        f"When other agents cannot reach the position of a higher surface like a high table or platform, quadrotors can assist in the transportation task. "
                        f"The quadrotor can fly across rooms, but only if the door between rooms is open. Only after the quadrotor has landed on a surface and is located in "
                        f"an area accessible to the robot dog or robot arm can the robot dog or robot arm pick up or place objects from the quadrotor's basket. "
                        f"The quadrotor itself does not have the ability to pick or place objects. If quadrotor is instructed to land on another surface, it need to take off, "
                        f"then movetowards the target position, and finally land. Quadrotor need to do it step by step, and can't skip one step.\n"
                    )
                    agent_rules.append(
                        "Quadrotor Rules:\n"
                        "1. If quadrotor is instructed to land on another surface, it need to take off, then movetowards the target position, and finally land. Quadrotor need to do it step by step, and can't skip one step.\n"
                        "2. The quadrotor can land on the same surface it just took off from, but it cannot execute a landing action immediately after the takeoff action because it lacks a movetowards action.\n"
                        "3. Quadrotor can only land on SURFACES with LANDABLE properties, otherwise it is not allowed.\n"
                        "4. The quadrotor's basket can transport multiple objects at the same time, which is used to improve efficiency when transporting multiple objects.\n"
                        "5. Before each step, you need to prioritize whether the task requires quadrotor's participation in the execution, and priority is given to satisfying the preconditions of the quadrotor action realization."
                        "6. When given preconditions for quadrotor action realization, make sure the conditions are met before the quadrotor can execute the action."
                        "7. If the quadrotor has not received the object, it should not take off to send the object.",
                    )
                elif agent['class_name'] == 'robot dog':
                    agent_capabilities.append(
                        f"robot dog: The robot dog has a robotic arm installed on its back, so it can be used to carry and operate objects located on a lower surface, "
                        f"or objects on the floor. But the robot dog cannot reach the position of a high surface. In addition, the robot dog and its robotic arm can also "
                        f"be used to open and close doors and other accessible containers only when it is close to the door or container after movetowards it. "
                        f"The robot dog can move across rooms, but only if the doors between them are opened. So the robot dog can first help other agents or itself open the door. "
                        f"When the robot dog's robot arm is holding something, it is not allowed to open and close doors or open and close containers. "
                        f"The robot dog needs to get close to the object before performing the operation. If the object is on top of the surface or inside a container,"
                        f"the robot dog can choose to move directly towards the object instead of movetowards the surface/container.\n"
                    )
                    agent_rules.append(
                        "Robot Dog Rules:\n"
                        "1. When an object is located on a surface or inside a container, robot dog can operate on it by moving directly towards the object rather than towards the surface or container.\n"
                        "2. When the robot dog wants to grab an object in the quadrotor basket or put an object into the basket, the robot dog should movetowards <basket> instead of movetowards <quadrotor>.\n"
                        "3. For being CLOSE to one object, robot dog has to movetowards the object before it can perform the next action.\n"
                        "4. Objects in the robot dog's hand are not allowed to be placed on the floor.\n"
                        "5. If the robot dog wants to open a door or other container, it needs to free its hand first, so it needs to put the object on a nearby table that can be touched.\n"
                        "6. The robot dog does not need to movetowards the door when entering another room, it can movetowards the room directly.\n"
                        "7. The surface of \"LOW_HEIGHT\" is something the robot dog can touch. The surface of \"HIGH_HEIGHT\" and the object of \"ON_HIGH_SURFACE\" are inaccessible to the robot dog.\n"
                        "8. If the robot dog needs to transport an object to another room:\n"
                        "   - Opening the door is a higher priority than grabbing the object (need empty hands for door)\n"
                        "   - Walk to the door that leads to the other room\n"
                        "   - Open the door\n"
                        "   - Walk to the object that needs to be transported\n"
                        "   - Grab it\n"
                        "   - Walk to the other room"
                    )
                elif agent['class_name'] == 'robot arm':
                    agent_capabilities.append(
                        f"robot arm: Different from the robot arm of the robot dog, the robot arm is fixed on a table or platform and is used to operate objects on the surface. "
                        f"The arm can be used to pick and place objects on the table, open or close containers on the table, or pick up or place objects from the basket of "
                        f"the quadrotor that lands on the table. Objects on other tables that the robot arm can't touch. If the drone lands on a different table than the robot arm, "
                        f"the robot arm is also out of reach. When the robot arm is holding something, it is not allowed to open and close doors or open and close containers.\n"
                    )
                    agent_rules.append(
                        "Robot Arm Rules:\n"
                        "1. The robot arm can only operate objects on its own table.\n"
                        "2. When the robot arm is holding something, it is not allowed to open and close containers."
                    )
            
            # Add group context after task goal
            group_agents_desc = [f"<{agent['class_name']}>({agent['id']})" for agent in group_info['agents']]
            group_agents_text = ", ".join(group_agents_desc)
            group_context = f"\nCurrent planning is for Group {group_info['group_id']} containing agents: {group_agents_text}\n"
            group_context += f"Sub-goal for this group: {group_info['sub_goal']}\n"
            
            oracle_prompt = oracle_prompt.replace('#TASK_GOAL#', f"{goal_instruction}\n")
            oracle_prompt = oracle_prompt.replace('#GROUP_INFO#', group_context)
        else:
            # If no group info, include all agent capabilities and rules
            agent_capabilities = [
                "quadrotor: The quadrotor can take off, land, and fly in the air. The quadrotor has a basket that can be used to transport objects. "
                "When other agents cannot reach the position of a higher surface like a high table or platform, quadrotors can assist in the transportation task. "
                "The quadrotor can fly across rooms, but only if the door between rooms is open. Only after the quadrotor has landed on a surface and is located in "
                "an area accessible to the robot dog or robot arm can the robot dog or robot arm pick up or place objects from the quadrotor's basket. "
                "The quadrotor itself does not have the ability to pick or place objects. If quadrotor is instructed to land on another surface, it need to take off, "
                "then movetowards the target position, and finally land. Quadrotor need to do it step by step, and can't skip one step.",
                
                "robot dog: The robot dog has a robotic arm installed on its back, so it can be used to carry and operate objects located on a lower surface, "
                "or objects on the floor. But the robot dog cannot reach the position of a high surface. In addition, the robot dog and its robotic arm can also "
                "be used to open and close doors and other accessible containers only when it is close to the door or container after movetowards it. "
                "The robot dog can move across rooms, but only if the doors between them are opened. So the robot dog can first help other agents or itself open the door. "
                "When the robot dog's robot arm is holding something, it is not allowed to open and close doors or open and close containers. "
                "The robot dog needs to get close to the object before performing the operation. If the object is on top of the surface or inside a container,"
                "the robot dog can choose to move directly towards the object instead of movetowards the surface/container.",
                
                "robot arm: Different from the robot arm of the robot dog, the robot arm is fixed on a table or platform and is used to operate objects on the surface. "
                "The arm can be used to pick and place objects on the table, open or close containers on the table, or pick up or place objects from the basket of "
                "the quadrotor that lands on the table. Objects on other tables that the robot arm can't touch. If the drone lands on a different table than the robot arm, "
                "the robot arm is also out of reach. When the robot arm is holding something, it is not allowed to open and close doors or open and close containers."
            ]

            # Add agent-specific rules
            agent_rules = [
                "Quadrotor Rules:\n"
                "1. If quadrotor is instructed to land on another surface, it need to take off, then movetowards the target position, and finally land. Quadrotor need to do it step by step, and can't skip one step.\n"
                "2. The quadrotor can land on the same surface it just took off from, but it cannot execute a landing action immediately after the takeoff action because it lacks a movetowards action.\n"
                "3. Quadrotor can only land on SURFACES with LANDABLE properties, otherwise it is not allowed.\n"
                "4. The quadrotor's basket can transport multiple objects at the same time, which is used to improve efficiency when transporting multiple objects.\n"
                "5. Before each step, you need to prioritize whether the task requires quadrotor's participation in the execution, and priority is given to satisfying the preconditions of the quadrotor action realization."
                "6. When given preconditions for quadrotor action realization, make sure the conditions are met before the quadrotor can execute the action."
                "7. If the quadrotor has not received the object, it should not take off to send the object.",

                "Robot Dog Rules:\n"
                "1. When an object is located on a surface or inside a container, robot dog can operate on it by moving directly towards the object rather than towards the surface or container.\n"
                "2. When the robot dog wants to grab an object in the quadrotor basket or put an object into the basket, the robot dog should movetowards <basket> instead of movetowards <quadrotor>.\n"
                "3. For being CLOSE to one object, robot dog has to movetowards the object before it can perform the next action.\n"
                "4. Objects in the robot dog's hand are not allowed to be placed on the floor.\n"
                "5. If the robot dog wants to open a door or other container, it needs to free its hand first, so it needs to put the object on a nearby table that can be touched.\n"
                "6. The robot dog does not need to movetowards the door when entering another room, it can movetowards the room directly.\n"
                "7. The surface of \"LOW_HEIGHT\" is something the robot dog can touch. The surface of \"HIGH_HEIGHT\" and the object of \"ON_HIGH_SURFACE\" are inaccessible to the robot dog.\n"
                "8. If the robot dog needs to transport an object to another room:\n"
                "   - Opening the door is a higher priority than grabbing the object (need empty hands for door)\n"
                "   - Walk to the door that leads to the other room\n"
                "   - Open the door\n"
                "   - Walk to the object that needs to be transported\n"
                "   - Grab it\n"
                "   - Walk to the other room",

                "Robot Arm Rules:\n"
                "1. The robot arm can only operate objects on its own table.\n"
                "2. When the robot arm is holding something, it is not allowed to open and close containers.",

                "Multi-Agent Cooperation Rules:\n"
                "1. If the robot dog and arm can do the task, it doesn't have to involve quadrotor.\n"
                "2. Because the robot dog cannot touch the high_surface, it needs to be assisted by the quadrotor.\n"
                "3. When quadrotor needs to interact with a robot dog:\n"
                "   - The quadrotor must land on the low_surface\n"
                "   - The robot dog can then movetowards the quadrotor\n"
                "   - The robot dog can put objects in the basket or grab objects from the basket\n"
                "4. For cross-room operations:\n"
                "   - If the door is closed, the robot dog is needed to open the door\n"
                "   - The robot dog must have empty hands to open the door\n"
                "   - Let the quadrotor fly in after the door is opened\n"
                "   - The robot dog can then perform object manipulation after quadrotor lands"
            ]
            oracle_prompt = oracle_prompt.replace('#TASK_GOAL#', goal_instruction)
            # Remove GROUP_INFO placeholder when no group info is provided
            oracle_prompt = oracle_prompt.replace('#GROUP_INFO#', "")

        agent_capabilities_text = "\n\n".join(agent_capabilities)
        oracle_prompt = oracle_prompt.replace('#AGENT_CAPABILITIES#', agent_capabilities_text)
        agent_rules_text = "\n\n".join(agent_rules)
        oracle_prompt = oracle_prompt.replace('#AGENT_RULES#', agent_rules_text)
        oracle_prompt = oracle_prompt.replace('#AGENT_OBSERVATIONS#', obs_text)
        oracle_prompt = oracle_prompt.replace('#DIALOGUE_HISTORY#', dialogue_history)
        oracle_prompt = oracle_prompt.replace('#NUMBER_AGENTS#', str(num_agents))
        
        return oracle_prompt
    
    def oracle_planning_vanilla(
            self,
            obs_text,
            goal_instruction,
            num_agents,
            dialogue_history=None,  # Make dialogue_history optional and deprecated
            group_info=None,  # Add group_info parameter
        ):
        '''
        Doing vanilla oracle planning, which is the process of the oracle agent generating the plan for the task.
        Args:
            obs_text: str, the text of the observation of the agents
            goal_instruction: str, the goal instruction of the task
            num_agents: int, the number of agents
            dialogue_history: str, the dialogue history (deprecated, will be removed in future versions)
            group_info: dict, optional information about the group being planned for
        Returns:
            message: str, the message of the oracle agent
            usage: int, the usage of the llm engine
        '''
        # import the structured prompt and insert the information into the prompt
        oracle_prompt = self.get_oracle_prompt(obs_text, goal_instruction, num_agents, dialogue_history, group_info)
      
        chat_prompt = [{"role": "user", "content": oracle_prompt}]
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        message = outputs[0]

        return message, usage

    def extract_structured_message(self, message):
        '''
        Extract the structured message from the message from the vanilla planning of the oracle agent
        '''
        extract_prompt = message + '\n' + \
            'Extract from the above paragraph the content of the format "Hello <class name>(id): message.". ' + \
            'Then output the contents of this section. Be careful not to output any superfluous content, exactly in the format given. ' + \
            'If the above paragraph is not exactly formatted as "Hello <class name>(id): #message#.", output similar content in this format. ' + \
            'As an example, the output might read: "Hello <robot dog>(0): please movetowards the <door>(1), and then open the <door>(1)". ' + \
            'If this format does not appear in the preceding text, please summarize the above content into this format for output. ' + \
            'To emphasize once again, the names of all objects and agent robots must be enclosed in <>, and the (id) must not be omitted. ' + \
            'Class name missing <> and (id) should be completed with these elements. ' + \
            'Please strictly follow this format in the output content.' 
        
        chat_prompt = [{"role": "user", "content": extract_prompt}]
        outputs, usage = self.generator(chat_prompt , self.sampling_params)
        message_output = outputs[0]
        
        return message_output, usage

    def get_agent_selection_prompt(self, agents_info_text, goal_instruction, obs_text, dialogue_history):
        '''
        Get the agent selection prompt
        '''
        with open(self.agent_selection_prompt_path, 'r') as f:
            agent_selection_prompt = f.read()
        agent_selection_prompt = agent_selection_prompt.replace('#AGENTS_INFO#', agents_info_text)
        agent_selection_prompt = agent_selection_prompt.replace('#TASK_GOAL#', goal_instruction)
        agent_selection_prompt = agent_selection_prompt.replace('#OBSERVATIONS#', obs_text)
        agent_selection_prompt = agent_selection_prompt.replace('#DIALOGUE_HISTORY#', dialogue_history)

        return agent_selection_prompt
    
    def agent_selection(
            self,
            agents_info_text,
            goal_instruction,
            obs_text,
            dialogue_history,
        ):
        '''
        Perform agent selection using LLM to choose the most suitable agents for the task.
        Args:
            agents_info_text: str, formatted text containing information about available agents
            goal_instruction: str, the goal instruction of the task
            obs_text: str, the text of the observation of the environment
            dialogue_history: str, the dialogue history
        Returns:
            selection_output: str, the raw output from LLM containing selected agent indices
            usage: float, the cost of the llm engine
        '''
        # Get the agent selection prompt
        agent_selection_prompt = self.get_agent_selection_prompt(agents_info_text, goal_instruction, obs_text, dialogue_history)
        
        chat_prompt = [{"role": "user", "content": agent_selection_prompt}]
        outputs, usage = self.generator(chat_prompt, self.sampling_params)
        selection_output = outputs[0]

        return selection_output, usage


