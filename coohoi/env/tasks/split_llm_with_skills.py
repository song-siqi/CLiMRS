import re
text = """3. **Search Workflow**:
   - **Step 1**: Use <wheeled robot1> (202) to explore area <A> (001) for the <trunk> (303). 
     - Skill: A
   - **Step 2**: Use <wheeled robot2> (203) to explore area <B> (002) for the <left wheel> (405).
     - Skill: B
   - **Step 3**: Use <wheeled robot3> (204) to explore area <D> (004) for the <right wheel> (406).
     - Skill: D
"""
def parse_workflow_text(text):
    import re
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


