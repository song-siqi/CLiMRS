
import copy
import openai

import json
from openai import OpenAIError, OpenAI
import backoff

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from llm_utils.llm_module import Agent, API_KEY_R17B, API_KEY_CLIMRS, API_URL, API_URL_R17B, MODEL_SELECTION

from types import SimpleNamespace


class LLM:
	def __init__(self, source, lm_id, args):

		self.args = args
		self.debug = args.debug
		self.source = args.source
		self.lm_id = args.lm_id
		self.chat = True
		self.total_cost = 0
		self.device = None
		self.record_dir = f'./log/{args.env}.txt'

		if self.source == 'openai':

			api_key = args.api_key  # your openai api key
			organization= args.organization # your openai organization

			client = OpenAI(api_key = api_key, organization=organization)
			if self.chat:
				self.sampling_params = {
					"max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": 1.0,
                    "n": args.n
				}
		
		elif self.source == 'llm_module':

			api_key = API_KEY_CLIMRS
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
			# @backoff.on_exception(backoff.expo, OpenAIError)
			@backoff.on_exception(backoff.expo, Exception)
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

	def parse_answer(self, available_actions, text):
		
		text = text.replace("_", " ")
		text = text.replace("takeoff from", "takeoff_from")
		text = text.replace("land on", "land_on")

		for i in range(len(available_actions)):
			action = available_actions[i]
			if action in text:
				return action
		self.write_log_to_file('\nThe first action parsing failed!!!')

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
				return action
		self.write_log_to_file('\nThe second action parsing failed!!!')

		print("WARNING! No available action parsed!!! Output plan NONE!\n")
		return None

	def _get_available_plans_with_params(self, agent_node, next_rooms, all_landable_surfaces, landable_surfaces, on_surfaces, 
						 grabbed_objects, reached_objects, unreached_objecs, on_same_surface_objects
						 ):
		"""
		New skill-based action format:
		
		'humanoid':
		[walk] <humanoid> (101) move to selected area
		[carry] <humanoid> (101) carry <obstacles> (507)
		[wait] <humanoid> (101) wait

		'wheeled robot':
		[move] <wheeled robot1/2/3> (202/203/204) move to component location using RRT path
		[push] <wheeled robot1/2/3> (202/203/204) push selected component to franka area
		[wait] <wheeled robot1/2/3> (202/203/204) wait

		'franka':
		[check] <franka> (606) check <trunk/left wheel/right wheel> (303/405/406)
		[pick] <franka> (606) pick and place <wheel> on <trunk>
		[wait] <franka> (606) wait

		'observation':
		[observe] area <A/B/C/D> (001-004) - only observe, robot doesn't move
		"""
		available_plans = []
		
		if agent_node["class_name"] == "quadrotor":
			other_landable_surfaces = []
			if "FLYING" in agent_node["states"]:
				if landable_surfaces is not None:
					available_plans.append(f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})")
					all_landable_surfaces.remove(landable_surfaces)
					other_landable_surfaces = copy.deepcopy(all_landable_surfaces)
				if len(other_landable_surfaces) != 0:
					for surface in other_landable_surfaces :
						available_plans.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
				for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

			if "LAND" in agent_node["states"]:
				if on_surfaces is not None:
					available_plans.append(f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})")

		elif agent_node["class_name"] == "humanoid":
			# New skill-based format for humanoid
			# Add walk skill to different areas
			for next_room in next_rooms:
				if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
					available_plans.append(f"[walk] <humanoid> (101) move to selected area")
			
			# Add carry skill for obstacles
			if len(unreached_objecs) != 0:
				for unreached_object in unreached_objecs:
					if unreached_object['class_name'] == 'obstacles':
						available_plans.append(f"[carry] <humanoid> (101) carry <obstacles> (507)")
			
			# Always add wait skill
			available_plans.append(f"[wait] <humanoid> (101) wait")

		elif agent_node["class_name"] in ["robot dog", "robot_dog", "mobile_car", "wheeled robot", "wheeled robot1", "wheeled robot2", "wheeled robot3"]:
			# New skill-based format for wheeled robots
			agent_id = str(agent_node.get('id', 202))  # Default to 202 if no ID, ensure string
			robot_name = "mobile_car"
			robot_id = "202"
			
			# Add move skills to component locations
			if len(unreached_objecs) != 0:
				for unreached_object in unreached_objecs:
					if unreached_object['class_name'] in ['trunk', 'left wheel', 'right wheel']:
						available_plans.append(f"[move] <{robot_name}> ({robot_id}) move to component location using RRT path")
			
			# Add push skills for components
			if len(reached_objects) != 0:
				for reached_object in reached_objects:
					if reached_object['class_name'] in ['trunk', 'left wheel', 'right wheel']:
						available_plans.append(f"[push] <{robot_name}> ({robot_id}) push selected component to franka area")
			
			# Always add wait skill
			available_plans.append(f"[wait] <{robot_name}> ({robot_id}) wait")

		elif agent_node['class_name'] in ['robot arm', 'robot_arm', 'franka']:
			# New skill-based format for franka robot arm
			
			# Add check skills for components
			for on_same_surface_object in on_same_surface_objects:
				if on_same_surface_object['class_name'] == 'trunk':
					available_plans.append(f"[check] <franka> (606) check <trunk> (303)")
				elif on_same_surface_object['class_name'] == 'left wheel':
					available_plans.append(f"[check] <franka> (606) check <left wheel> (405)")
				elif on_same_surface_object['class_name'] == 'right wheel':
					available_plans.append(f"[check] <franka> (606) check <right wheel> (406)")
			
			# Add pick and place skills if components are available
			trunk_available = any(obj['class_name'] == 'trunk' for obj in on_same_surface_objects)
			left_wheel_available = any(obj['class_name'] == 'left wheel' for obj in on_same_surface_objects)
			right_wheel_available = any(obj['class_name'] == 'right wheel' for obj in on_same_surface_objects)
			
			if trunk_available and left_wheel_available:
				available_plans.append(f"[pick] <franka> (606) pick and place <left wheel> (405) on <trunk> (303)")
			if trunk_available and right_wheel_available:
				available_plans.append(f"[pick] <franka> (606) pick and place <right wheel> (406) on <trunk> (303)")
			
			# Always add wait skill
			available_plans.append(f"[wait] <franka> (606) wait")

		# Add observation skills for all agents (environmental awareness)
		if not available_plans or len(available_plans) == 1:  # If only wait skill or no skills
			available_plans.extend([
				"[observe] area <A> (001) - only observe, robot doesn't move",
				"[observe] area <B> (002) - only observe, robot doesn't move", 
				"[observe] area <C> (003) - only observe, robot doesn't move",
				"[observe] area <D> (004) - only observe, robot doesn't move"
			])

		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"
		print(agent_node["class_name"],agent_node['id'])
		print(available_plans)
		return plans, len(available_plans), available_plans

		
	def run(self, agent_node, chat_agent_info,current_room, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects):
		info = {"num_available_actions": None,
			"prompts": None,
			"outputs": None,
			"plan": None,
			"action_list": None,
			"cost":self.total_cost, 
			f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost}

		prompt_path = chat_agent_info['prompt_path']
		with open(prompt_path, 'r') as f:
			agent_prompt = f.read()

		available_plans, num, available_plans_list = self._get_available_plans_with_params(agent_node, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects,
																		 )
		
		agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
		agent_prompt = agent_prompt.replace('#ACTIONLIST#', available_plans)
		agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
		
		if self.debug:
			print(f"cot_prompt:\n{agent_prompt}")
		chat_prompt = [{"role": "user", "content": agent_prompt}]
		outputs, usage = self.generator(chat_prompt, self.sampling_params)
		output = outputs[0]

		self.write_log_to_file(output+'\n111111111')
		self.total_cost += usage
		info['cot_outputs'] = outputs

		if self.debug:
			print(f"cot_output:\n{output}")
			print(f"total cost: {self.total_cost}")
		sentences = output.split(".")
		first_sentence = sentences[0].upper()
		print("#" *20)
		print("the first sentence is", first_sentence)
		print("#" *20)

		if first_sentence == "YES I CAN":
			chat_prompt = [{"role": "user", "content": agent_prompt},
							{"role": "assistant", "content": output},
							{"role": "user", "content": "Answer with only one best next action in the list of available actions. So the answer is"}]

			outputs, usage = self.generator(chat_prompt, self.sampling_params)
			output = outputs[0]
			self.total_cost += usage
			self.write_log_to_file(output+'\n2222222222222')
			sentences = output.split(".")
			first_sentence = sentences[0].upper()
			if first_sentence != "SORRY I CANNOT": 

				if self.debug:
					print(f"cot_output:\n{output}")
					print(f"total cost: {self.total_cost}")

				plan = self.parse_answer(available_plans_list, output)
				if plan is None:
					plan_str = 'no plan'
				else:
					plan_str = plan
				print(plan)
				if self.debug:
					print(f"plan: {plan}\n")
				info.update({"num_available_actions": num,
						"prompts": chat_prompt,
						# "outputs": outputs,
						"plan": plan,
						"action_list": available_plans_list,
						f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost})
				message = f" The action I finally decided to perform is {plan_str}. "

				prompt_path = self.args.judge_prompt_path
				with open(prompt_path, 'r') as f:
					prompt = f.read()
				prompt = prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
				prompt = prompt.replace('#PLAN#', plan_str)
				prompt = prompt.replace('#AGENT#', f"<{agent_node['class_name']}>")
				prompt = [{"role": "user", "content": prompt}]
				outputs, usage = self.generator(prompt, self.sampling_params)
				output = outputs[0]
				self.total_cost += usage
				message += output
				self.write_log_to_file(output+'\n333333333333333333')
				info.update({"outputs": message})


		if first_sentence == "SORRY I CANNOT":
			output = output[16].lower() + output[17:]
			message = f"Sorry, the current actions I can perform cannot complete this instrcution. Possible reasons would be {output} My current actionlist is: {available_plans}"
			self.write_log_to_file(message+'\n4444444444444')
		info['cost'] = self.total_cost	
		self.write_log_to_file(f"total cost: {self.total_cost}")
		info.update({"outputs": message})
		return message, info

	def write_log_to_file(self,log_message, file_name=None):
		file_name = self.record_dir
		with open(file_name, 'a') as file:  
			file.write(log_message + '\n')  