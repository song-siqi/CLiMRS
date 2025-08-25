import requests
import json

# 1. 配置 API 信息
API_URL = "http://35.220.164.252:3888/v1/chat/completions"  # 替换为你的 API URL
API_KEY = "sk-VS6OzWyx7SyeeNnWRo7BuUeD9H9jzxU88z9IQlcf4K72l14U"  # 替换为你的 API 密钥
# base_url = ""


# 2. 定义一个函数来调用 API
def ask_llm(question):
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
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
                State Description:
                unknown area: <A> (001), <B> (002), <C> (003), <D> (004), where A,B,C,D represent the first to the fourth quadrants respectively.
                component list: <trunk> (303), <left wheel> (405), <right wheel> (406), <obstacles> (507).
                agent list: <wheeled robot1> (202),<wheeled robot2> (203),<wheeled robot3> (204), <humanoid> (101), <franka> (606).
                init state: <wheeled robot1> (202) is close to <A> (001), <wheeled robot2> (203) is close to <B> (002), <wheeled robot3> (204) is close to <D> (004),
                            <humanoid> (101) is (0.0, 0.0), <franka> (606) is fixed in (0.0, -2.0).
                Skill list:
                A. [explore] area <A> (001)
                B. [explore] area <B> (002)
                C. [explore] area <C> (003)
                D. [explore] area <D> (004)
                E. [move] <wheeled robot1> (202) move to selected area
                F. [move] <wheeled robot2> (203) move to selected area
                G. [move] <wheeled robot3> (204) move to selected area
                H. [walk] <humanoid> (101) move to selected area
                I. [carry] <humanoid> (101) carry <obstacles> (507)
                J. [push] <wheeled robot1> (202) push selected component
                K. [push] <wheeled robot2> (203) push selected component
                L. [push] <wheeled robot3> (204) push selected component
                M. [check] <franka> (606) check <trunk> (303)
                N. [check] <franka> (606) check <left wheel> (405)
                O. [check] <franka> (606) check <right wheel> (406)
                P. [pick] <franka> (606) pick <left wheel> (405)
                Q. [pick] <franka> (606) pick <right wheel> (406)
                R. [place] <franka> (606) place <left wheel> (405) on <trunk> (303)
                S. [place] <franka> (606) place <right wheel> (406) on <trunk> (303) 
                T. [wait] <franka> (606) wait
                U. [wait] <humanoid> (101) wait
                V. [wait] <wheeled robot1> (202) wait
                W. [wait] <wheeled robot2> (203) wait
                X. [wait] <wheeled robot3> (204) wait
                Answer: Let's think step by step.
            """
    
    payload = json.dumps({
        "model":"gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt.format(question)}
        ]
    })

    url = API_URL
    try:
        response = requests.post(url, headers=headers, data=payload)
        response = response.json()
        return response.get("choices", [])[0].get("message", {}).get("content", "")
        # response.raise_for_status()  # 检查是否有错误
        # result = response.json()  # 解析返回的 JSON 数据
        # return result.get("choices")[0].get("text", "").strip()  # 获取模型返回的文本
    except Exception as e:
        return f"Error: {e}"

# 3. 主程序逻辑
if __name__ == "__main__":
    print("欢迎使用问答系统！（输入 'exit' 退出）")
    while True:
        user_input = input()
        if user_input.lower() == "exit":
            print("再见！")
            break
        answer = ask_llm(user_input)
        print(f"AI 的回答：{answer}")
