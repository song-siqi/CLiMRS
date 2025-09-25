from typing import Optional
from .ask_Llm import LLMWorkflow

class LLMObserver:
    def __init__(self):
        self.llm: Optional[LLMWorkflow] = None
        self.task = None
        
    def set_task(self, task):
        """Set the task/environment to observe"""
        self.task = task
    
    def update(self):
        """Called when the environment state changes"""
        if self.task is None:
            return
            
        env_id = 0
        env_ptr = self.task.envs[0] if hasattr(self.task, 'envs') else None
        
        area_positions, agent_positions = self.task.get_positions_for_prompt(env_id, env_ptr)
        actor_indices_map = {}

        if self.llm is None:
            self.llm = LLMWorkflow(area_positions, agent_positions, actor_indices_map)
        else:
            self.llm.area_positions = area_positions
            self.llm.agent_positions = agent_positions
            
        answer = self.llm.ask_llm("Please provide assembly plan")
        return answer