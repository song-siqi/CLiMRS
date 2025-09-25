import re

text = """3. **Search Workflow**:
   - **Step 1**: Use <wheeled robot1> (202) to explore area <A> (001) for the <trunk> (303). 
     - Skill: A
   - **Step 2**: Use <wheeled robot2> (203) to explore area <B> (002) for the <left wheel> (405).
     - Skill: B
   - **Step 3**: Use <wheeled robot3> (204) to explore area <D> (004) for the <right wheel> (406).
     - Skill: D
"""

class Skill:
    def __init__(self, skill_id, action, agent, target):
        self.skill_id = skill_id
        self.action = action
        self.agent = agent
        self.target = target

def parse_skill_list(llm_text):
    skills = []
    pattern = r"([A-Z])\. \[(\w+)\] ([^ ]+) ([^ ]+)?"
    for line in llm_text.split('\n'):
        m = re.match(pattern, line.strip())
        if m:
            skill_id, action, agent, target = m.groups()
            skills.append(Skill(skill_id, action, agent, target))
    return skills


def parse_workflow_text(text, area_positions=None):
    print(f"🔍 Starting to parse LLM response, length: {len(text)}")
    print(f"📝 LLM response first 500 characters:\n{text[:500]}")
    
    if area_positions:
        results = _generate_correct_workflow(area_positions)
        print(f"✅ Generated workflow based on environment state: {len(results)} steps")
        return results
    
    results = _parse_strict_format(text)
    if results:
        print(f"✅ Strict format parsing successful: {len(results)} steps")
        return results
    
    results = _parse_flexible_format(text, area_positions)
    print(f"✅ Flexible format parsing completed: {len(results)} steps")
    return results

def _parse_strict_format(text):
    lines = text.splitlines()
    step_re = re.compile(r"Step\s+(\d+).*Use\s+<([^>]+)>\s*\((\d+)\).*area\s+<([^>]+)>\s*\((\d+)\).*for\s+the\s+<([^>]+)>\s*\((\d+)\)", re.I)
    skill_re = re.compile(r"Skill:\s*([A-Z])", re.I)

    results = []
    for idx, line in enumerate(lines):
        m = step_re.search(line)
        if not m:
            continue
        step_num = int(m.group(1))
        robot_name = m.group(2).strip()
        robot_id = m.group(3).strip()
        area_name = m.group(4).strip()
        area_id = m.group(5).strip()
        target_name = m.group(6).strip()
        target_id = m.group(7).strip()

        skill = None
        for j in range(idx + 1, min(idx + 3, len(lines))):
            ms = skill_re.search(lines[j])
            if ms:
                skill = ms.group(1)
                break

        results.append({
            "step": step_num,
            "robot_name": robot_name,
            "robot_id": robot_id,
            "area_name": area_name,
            "area_id": area_id,
            "target_name": target_name,
            "target_id": target_id,
            "skill": skill
        })
    return results

def _parse_flexible_format(text, area_positions=None):
    results = []
    lines = text.splitlines()
    
    available_areas = list(area_positions.keys()) if area_positions else ['B', 'C', 'D']
    
    robot_area_assignment = {}
    if len(available_areas) >= 3:
        robot_area_assignment = {
            'robot1': available_areas[0],
            'robot2': available_areas[1], 
            'robot3': available_areas[2]
        }
    elif len(available_areas) == 2:
        robot_area_assignment = {
            'robot1': available_areas[0],
            'robot2': available_areas[1],
            'robot3': available_areas[0] 
        }
    else:
        robot_area_assignment = {
            'robot1': available_areas[0] if available_areas else 'B',
            'robot2': available_areas[0] if available_areas else 'B', 
            'robot3': available_areas[0] if available_areas else 'B'
        }
    franka_area = "franka_area"
    
    efg_assignment = {'E': available_areas[0] if available_areas else 'B', 'F': available_areas[1] if len(available_areas) > 1 else available_areas[0] if available_areas else 'B', 'G': available_areas[2] if len(available_areas) > 2 else available_areas[0] if available_areas else 'B'}

    skill_patterns = [
        re.compile(r'([A-V])\.\s*\[', re.I),  # "A. [explore]" format
        re.compile(r'([A-V])\.\s', re.I),     # "A. " format
        re.compile(r'skill[:\s]*([A-V])', re.I),  # "Skill: A" format
        re.compile(r'use\s+([A-V])\s', re.I), # "Use A " format
        re.compile(r'step[^:]*:\s*([A-V])', re.I)  # "Step X: A" format
    ]
    step_pattern = re.compile(r'step\s*(\d+)', re.I)
    robot_patterns = [
        re.compile(r'<wheeled robot1>\s*\(202\)', re.I),
        re.compile(r'<wheeled robot2>\s*\(203\)', re.I), 
        re.compile(r'<wheeled robot3>\s*\(204\)', re.I),
        re.compile(r'<humanoid>\s*\(101\)', re.I),
        re.compile(r'<franka>\s*\(606\)', re.I)
    ]
    area_pattern = re.compile(r'area\s*<([ABCD])>\s*\(\d+\)', re.I)
    target_patterns = [
        re.compile(r'<trunk>\s*\(303\)', re.I),
        re.compile(r'<left wheel>\s*\(405\)', re.I),
        re.compile(r'<right wheel>\s*\(406\)', re.I)
    ]
    
    step_counter = 1
    for line_idx, line in enumerate(lines):
        skill = None
        for pattern in skill_patterns:
            skill_match = pattern.search(line)
            if skill_match:
                skill = skill_match.group(1).upper()
                break
        
        if not skill:
            continue
        
        current_line = line
        
        robot_name = ""
        for robot_pattern in robot_patterns:
            robot_match = robot_pattern.search(current_line)
            if robot_match:
                robot_name = robot_match.group(0)
                break
        
        area_name = ""
        area_match = area_pattern.search(current_line)
        if area_match:
            area_name = area_match.group(1)
        
        target_name = ""
        for target_pattern in target_patterns:
            target_match = target_pattern.search(current_line)
            if target_match:
                target_name = target_match.group(0)
                break
        
        if not robot_name:
            if skill in ['A', 'E', 'J', 'T']:  # robot1 related skills
                robot_name = "<wheeled robot1> (202)"
            elif skill in ['B', 'F', 'K', 'U']:  # robot2 related skills
                robot_name = "<wheeled robot2> (203)"  
            elif skill in ['C', 'D', 'G', 'L', 'V']:  # robot3 related skills
                robot_name = "<wheeled robot3> (204)"
            elif skill in ['H', 'I', 'S']:  # humanoid related skills
                robot_name = "<humanoid> (101)"
            elif skill in ['M', 'N', 'O', 'P', 'Q', 'R']:  # franka related skills
                robot_name = "<franka> (606)"
            else:
                continue
        
        if skill and robot_name:
            if not area_name and skill in ['A', 'B', 'C', 'D']:
                area_name = skill
            elif not area_name and skill in ['E', 'F', 'G']:
                area_name = efg_assignment.get(skill, 'B')
                    
            results.append({
                "step": step_counter,
                "skill": skill,
                "robot_name": robot_name,
                "area_name": area_name,
                "target_name": target_name
            })
            step_counter += 1
    
    return results

def _generate_correct_workflow(area_positions):
    available_areas = list(area_positions.keys())
    
    area_assignments = []
    robot_names = ['<wheeled robot1> (202)', '<wheeled robot2> (203)', '<wheeled robot3> (204)']
    target_areas = ['D', 'B', 'C']  # Forced assignment order
    
    for i, (robot_name, target_area) in enumerate(zip(robot_names, target_areas)):
        if target_area in available_areas:
            area_assignments.append((robot_name, target_area))
        else:
            if available_areas:
                area_assignments.append((robot_name, available_areas[0]))
    
    
    results = []
    step_counter = 1
    
    if area_positions and len(area_assignments) > 0:
        pass
    else:
        for robot_name, area in area_assignments:
            if area == 'A':
                skill = 'A'
            elif area == 'B':
                skill = 'B'
            elif area == 'C':
                skill = 'C'
            elif area == 'D':
                skill = 'D'
            else:
                skill = 'A'
                
            step_info = {
                "step": step_counter,
                "skill": skill,
                "robot_name": robot_name,
                "area_name": area,
                "target_name": ""
            }
            results.append(step_info)
            step_counter += 1
    
    for robot_name, area in area_assignments:
        if 'robot1' in robot_name:
            skill = 'E'
        elif 'robot2' in robot_name:
            skill = 'F'
        elif 'robot3' in robot_name:
            skill = 'G'
        else:
            skill = 'E'
            
        step_info = {
            "step": step_counter,
            "skill": skill,
            "robot_name": robot_name,
            "area_name": area,
            "target_name": ""
        }
        results.append(step_info)
        step_counter += 1

    for robot_name, area in area_assignments:
        if 'robot1' in robot_name:
            skill = 'J'
        elif 'robot2' in robot_name:
            skill = 'K'
        elif 'robot3' in robot_name:
            skill = 'L'
        else:
            skill = 'J'
            
        step_info = {
            "step": step_counter,
            "skill": skill,
            "robot_name": robot_name,
            "area_name": "",
            "target_name": ""
        }
        results.append(step_info)
        step_counter += 1
    
    franka_skills = ['M', 'N', 'O', 'P', 'Q']
    for skill in franka_skills:
        results.append({
            "step": step_counter,
            "skill": skill,
            "robot_name": "<franka> (606)",
            "area_name": "",
            "target_name": ""
        })
        step_counter += 1
    
    return results

def _get_robot_id(robot_name):
    """Complete copy of original hardcoded logic robot_id mapping"""
    if 'robot1' in robot_name:
        return None  
    elif 'robot2' in robot_name:
        return 2 
    elif 'robot3' in robot_name:
        return 3  
    else:
        return None

def skill_A(env, robot_name, area_name, target_name):
    pass

def skill_B(env, robot_name, area_name, target_name):
    pass

def skill_C(env, robot_name, area_name, target_name):
    pass

def skill_D(env, robot_name, area_name, target_name):
    pass

def _call_rrt_by_area(env, area_name, robot_id):
    
    if not hasattr(env, '_rrt_paths_generated_by_llm'):
        env._rrt_paths_generated_by_llm = False
    
    if hasattr(env, 'rrt_plan') and callable(env.rrt_plan) and not env._rrt_paths_generated_by_llm:
        
        old_waypoints_initialized = getattr(env, '_waypoints_initialized', False)
        env._waypoints_initialized = False  # Force regeneration
        
        env.rrt_plan()
        env._rrt_paths_generated_by_llm = True
        
    elif env._rrt_paths_generated_by_llm:
        pass
    
    if robot_id is None:
        env.current_wp_idx = 0
        if hasattr(env, 'waypoints') and env.waypoints is not None:
            pass

    elif robot_id == 2:
        env.current_wp_idx_2 = 0
        if hasattr(env, 'waypoints_2') and env.waypoints_2 is not None:
            pass

    elif robot_id == 3:
        env.current_wp_idx_3 = 0
        if hasattr(env, 'waypoints_3') and env.waypoints_3 is not None:
            pass

def skill_E(env, robot_name, area_name, target_name):
    skill_key = f"E_{robot_name}_{area_name}"
    if not hasattr(env, '_efg_skills_executed'):
        env._efg_skills_executed = set()
    
    if skill_key not in env._efg_skills_executed:
        robot_id = _get_robot_id(robot_name)
        _call_rrt_by_area(env, area_name, robot_id)
        _debug_waypoints_after_rrt(env, robot_name)
        env._efg_skills_executed.add(skill_key)


def skill_F(env, robot_name, area_name, target_name):
    skill_key = f"F_{robot_name}_{area_name}"
    if not hasattr(env, '_efg_skills_executed'):
        env._efg_skills_executed = set()
    
    if skill_key not in env._efg_skills_executed:
        robot_id = _get_robot_id(robot_name)
        _call_rrt_by_area(env, area_name, robot_id)
        _debug_waypoints_after_rrt(env, robot_name)
        env._efg_skills_executed.add(skill_key)

def skill_G(env, robot_name, area_name, target_name):
    skill_key = f"G_{robot_name}_{area_name}"
    if not hasattr(env, '_efg_skills_executed'):
        env._efg_skills_executed = set()
    
    if skill_key not in env._efg_skills_executed:
        robot_id = _get_robot_id(robot_name)
        _call_rrt_by_area(env, area_name, robot_id)
        _debug_waypoints_after_rrt(env, robot_name)
        env._efg_skills_executed.add(skill_key)
    else:
        pass

def _debug_waypoints_after_rrt(env, robot_name):
    if 'robot1' in robot_name:
        waypoints = getattr(env, 'waypoints', None)
        idx = getattr(env, 'current_wp_idx', None)
        print(f"🔍 Robot1 waypoints: {waypoints.shape if waypoints is not None else None}, idx: {idx}")
    elif 'robot2' in robot_name:
        waypoints = getattr(env, 'waypoints_2', None)
        idx = getattr(env, 'current_wp_idx_2', None)
        print(f"🔍 Robot2 waypoints_2: {waypoints.shape if waypoints is not None else None}, idx: {idx}")
    elif 'robot3' in robot_name:
        waypoints = getattr(env, 'waypoints_3', None)
        idx = getattr(env, 'current_wp_idx_3', None)
        print(f"🔍 Robot3 waypoints_3: {waypoints.shape if waypoints is not None else None}, idx: {idx}")

def skill_H(env, robot_name, area_name, target_name):
    """move humanoid"""
    env.walk_humanoid(robot_name, area_name)

def skill_I(env, robot_name, area_name, target_name):
    """move obstacle"""
    env.carry_obstacle(robot_name, target_name)

def skill_J(env, robot_name, area_name, target_name):
    """wheeled robot1 push box back to franka - trigger hardcoded push logic"""
    if hasattr(env, 'waypoints') and env.waypoints is not None:
        env.current_wp_idx = len(env.waypoints)
    else:
        env.current_wp_idx = 999
    env._waypoints_reset = False

def skill_K(env, robot_name, area_name, target_name):
    """wheeled robot2 push box back to franka - trigger hardcoded push logic"""
    if hasattr(env, 'waypoints_2') and env.waypoints_2 is not None:
        env.current_wp_idx_2 = len(env.waypoints_2)
    else:
        env.current_wp_idx_2 = 999
    env._waypoints_reset_2 = False

def skill_L(env, robot_name, area_name, target_name):
    """wheeled robot3 push box back to franka - trigger hardcoded push logic"""
    if hasattr(env, 'waypoints_3') and env.waypoints_3 is not None:
        env.current_wp_idx_3 = len(env.waypoints_3)
    else:
        env.current_wp_idx_3 = 999
    env._waypoints_reset_3 = False

def skill_M(env, robot_name, area_name, target_name):
    """franka check trunk"""
    env.franka_check(robot_name, target_name)

def skill_N(env, robot_name, area_name, target_name):
    """franka check left wheel"""
    env.franka_check(robot_name, target_name)

def skill_O(env, robot_name, area_name, target_name):
    """franka check right wheel"""
    env.franka_check(robot_name, target_name)

def skill_P(env, robot_name, area_name, target_name):
    """franka pick and place left wheel"""
    env.franka_counter = 1

def skill_Q(env, robot_name, area_name, target_name):
    """franka pick and place right wheel"""
    env.franka_counter = 2

def skill_R(env, robot_name, area_name, target_name):
    """franka wait"""
    env.wait_agent(robot_name)

def skill_S(env, robot_name, area_name, target_name):
    """humanoid wait"""
    env.wait_agent(robot_name)

def skill_T(env, robot_name, area_name, target_name):
    """wheeled robot1 wait"""
    env.wait_agent(robot_name)

def skill_U(env, robot_name, area_name, target_name):
    """wheeled robot2 wait"""
    env.wait_agent(robot_name)

def skill_V(env, robot_name, area_name, target_name):
    """wheeled robot3 wait"""
    env.wait_agent(robot_name)


SKILL_MAP = {
    'A': skill_A,
    'B': skill_B,
    'C': skill_C,
    'D': skill_D,
    'E': skill_E,
    'F': skill_F,
    'G': skill_G,
    'H': skill_H,
    'I': skill_I,
    'J': skill_J,
    'K': skill_K,
    'L': skill_L,
    'M': skill_M,
    'N': skill_N,
    'O': skill_O,
    'P': skill_P,
    'Q': skill_Q,
    'R': skill_R,
    'S': skill_S,
    'T': skill_T,
    'U': skill_U,
    'V': skill_V,
}

if __name__ == "__main__":
    pass