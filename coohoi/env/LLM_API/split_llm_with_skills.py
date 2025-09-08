import re
"""
This function parses the workflow text and extracts relevant information about each step.
"""
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


def parse_workflow_text(text):
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

### explore area ###
def skill_A(env, robot_name, area_name, target_name):
    """explore areaA"""
    env.explore_area(robot_name, area_name)

def skill_B(env, robot_name, area_name, target_name):
    """explore areaB"""
    env.move_robot(robot_name, area_name)

def skill_C(env, robot_name, area_name, target_name):
    """explore areaC"""
    env.explore_area(robot_name, area_name)

def skill_D(env, robot_name, area_name, target_name):
    """explore areaD"""
    env.explore_area(robot_name, area_name)

### robot move ###
def skill_E(env, robot_name, area_name, target_name):
    """wheeled robot1"""
    env.move_robot(robot_name, area_name)

def skill_F(env, robot_name, area_name, target_name):
    """wheeled robot2"""
    env.move_robot(robot_name, area_name)

def skill_G(env, robot_name, area_name, target_name):
    """wheeled robot3"""
    env.move_robot(robot_name, area_name)

### humanoid ###
def skill_H(env, robot_name, area_name, target_name):
    """move humanoid"""
    env.walk_humanoid(robot_name, area_name)

def skill_I(env, robot_name, area_name, target_name):
    """move obstacle"""
    env.carry_obstacle(robot_name, target_name)

def skill_J(env, robot_name, area_name, target_name):
    """wheeled robot1 push"""
    env.push_component(robot_name, target_name)

def skill_K(env, robot_name, area_name, target_name):
    """wheeled robot2 push"""
    env.push_component(robot_name, target_name)

def skill_L(env, robot_name, area_name, target_name):
    """wheeled robot3 push"""
    env.push_component(robot_name, target_name)

def skill_M(env, robot_name, area_name, target_name):
    """franka check trunk"""
    env.franka_check(robot_name, target_name)

def skill_N(env, robot_name, area_name, target_name):
    """franka检查左轮"""
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
    """franka等待"""
    env.wait_agent(robot_name)

def skill_S(env, robot_name, area_name, target_name):
    """humanoid等待"""
    env.wait_agent(robot_name)

def skill_T(env, robot_name, area_name, target_name):
    """wheeled robot1等待"""
    env.wait_agent(robot_name)

def skill_U(env, robot_name, area_name, target_name):
    """wheeled robot2等待"""
    env.wait_agent(robot_name)

def skill_V(env, robot_name, area_name, target_name):
    """wheeled robot3等待"""
    env.wait_agent(robot_name)


# skill mapping
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
    # 'W': skill_W,
    # 'X': skill_X,
}

# SKILL_MAP = {
#     'explore': lambda env, robot, area, target: env._explore_area(robot, area),
#     'move': lambda env, robot, area, target: env._move_to_area(robot, area),
#     'push': lambda env, robot, area, target: env._push_object(robot, target),
#     'carry': lambda env, robot, area, target: env._carry_object(robot, target),
#     'check': lambda env, robot, area, target: env._check_object(robot, target),
#     'wait': lambda env, robot, area, target: env._wait(robot)
# }