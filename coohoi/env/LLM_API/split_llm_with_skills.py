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

def skill_A(env, robot_name, area_name, target_name):
    """探索区域"""
    env.explore_area(robot_name, area_name)

def skill_B(env, robot_name, area_name, target_name):
    """移动到区域"""
    env.move_robot(robot_name, area_name)

def skill_C(env, robot_name, area_name, target_name):
    """探索另一区域"""
    env.explore_area(robot_name, area_name)

def skill_D(env, robot_name, area_name, target_name):
    """探索第四区域"""
    env.explore_area(robot_name, area_name)

def skill_E(env, robot_name, area_name, target_name):
    """移动wheeled robot1"""
    env.move_robot(robot_name, area_name)

def skill_F(env, robot_name, area_name, target_name):
    """移动wheeled robot2"""
    env.move_robot(robot_name, area_name)

def skill_G(env, robot_name, area_name, target_name):
    """移动wheeled robot3"""
    env.move_robot(robot_name, area_name)

def skill_H(env, robot_name, area_name, target_name):
    """移动humanoid"""
    env.walk_humanoid(robot_name, area_name)

def skill_I(env, robot_name, area_name, target_name):
    """搬运障碍物"""
    env.carry_obstacle(robot_name, target_name)

def skill_J(env, robot_name, area_name, target_name):
    """wheeled robot1推动组件"""
    env.push_component(robot_name, target_name)

def skill_K(env, robot_name, area_name, target_name):
    """wheeled robot2推动组件"""
    env.push_component(robot_name, target_name)

def skill_L(env, robot_name, area_name, target_name):
    """wheeled robot3推动组件"""
    env.push_component(robot_name, target_name)

def skill_M(env, robot_name, area_name, target_name):
    """franka检查trunk"""
    env.franka_check(robot_name, target_name)

def skill_N(env, robot_name, area_name, target_name):
    """franka检查左轮"""
    env.franka_check(robot_name, target_name)

def skill_O(env, robot_name, area_name, target_name):
    """franka检查右轮"""
    env.franka_check(robot_name, target_name)

def skill_P(env, robot_name, area_name, target_name):
    """franka抓取左轮"""
    env.franka_pick(robot_name, target_name)

def skill_Q(env, robot_name, area_name, target_name):
    """franka抓取右轮"""
    env.franka_pick(robot_name, target_name)

def skill_R(env, robot_name, area_name, target_name):
    """franka放置左轮到trunk上"""
    env.franka_place(robot_name, target_name)

def skill_S(env, robot_name, area_name, target_name):
    """franka放置右轮到trunk上"""
    env.franka_place(robot_name, target_name)

def skill_T(env, robot_name, area_name, target_name):
    """franka等待"""
    env.wait_agent(robot_name)

def skill_U(env, robot_name, area_name, target_name):
    """humanoid等待"""
    env.wait_agent(robot_name)

def skill_V(env, robot_name, area_name, target_name):
    """wheeled robot1等待"""
    env.wait_agent(robot_name)

def skill_W(env, robot_name, area_name, target_name):
    """wheeled robot2等待"""
    env.wait_agent(robot_name)

def skill_X(env, robot_name, area_name, target_name):
    """wheeled robot3等待"""
    env.wait_agent(robot_name)

# 技能映射字典
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
    'W': skill_W,
    'X': skill_X,
}
