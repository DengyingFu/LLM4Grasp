from langchain_openai import ChatOpenAI, OpenAI
from langgraph.graph import END, StateGraph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from Grasp import EmboidedGrasp
import re
from langgraph.graph.message import add_messages
import base64
import sys
# 创建一个tee函数
class Tee:
    def __init__(self, file):
        self.file = open(file, 'a')
        self.stdout = sys.stdout  # 保存原始stdout

    def write(self, message):
        self.file.write(message)  # 写入文件
        self.stdout.write(message)  # 同时写入控制台

    def flush(self):
        self.file.flush()
        self.stdout.flush()
sys.stdout = Tee('3.txt')


em_grasp = EmboidedGrasp()
def get_graspPosition():
    return em_grasp.get_grasp()

# def detect_by_text(text):
#     detections = em_grasp.detect_by_text('apple')
#     return detections

def filter_grasp_by_detections(text, grasp_pose):
    final_pose = em_grasp.filter_grasp_by_text(text,grasp_pose)
    return final_pose

def move_and_grasp(pose):
    em_grasp.execute_grasp(pose)

def get_put_positon(name):
    target_location = em_grasp.get_put(name)
    return target_location

def move_to(pose):
    em_grasp.move_to(pose)

def execute_reset():
    em_grasp.move_reset()

def move(direction, distance):
    em_grasp.move(direction, distance)

def ungrasp():
    em_grasp.ungrasp()


# 封装api，以供LLM调用
# def get_put_by_name(name_text):  # 获取需要放置的地方的矩阵
#     xyz1 = em_grasp.get_put(name_text)
#     return xyz1
#
#
# def execute_put(xyz1):
#     em_grasp.execute_putit(xyz1)
#
#
# def get_grasp_by_name(name_text):  # 获取抓取的姿态矩阵
#     print("name_text:", name_text)
#     grasp_poses = em_grasp.get_grasp(samples=100, vis=True)  # 得到samples个抓取姿态
#     grasp = em_grasp.filter_grasp_by_text(name_text, grasp_poses)  # 过滤
#     return grasp
#
#
# def execute_grasp(grasp):
#     em_grasp.execute_grasp(grasp)


"""
## 代码模板
def plan():
    try:
        # 步骤1：获取抓取目标
        grasp_target = get_grasp_by_name('[物品名称]')
        if not grasp_target:
            raise ValueError("未找到目标物体")

        # 步骤2：执行抓取
        execute_grasp(grasp_target[0])

        # 步骤3：获取放置位置
        place_position = get_put_by_name('[位置名称]')
        if not place_position.valid:
            raise ValueError("无效放置位置")

        # 步骤4：执行放置
        execute_put(place_position)
    except Exception as e:
        print(f"操作失败：{str(e)}")
        return False
    return True
"""
MAIN_SYSTEM_PROMPT = SystemMessage("""## 角色
您是一个多模态机械臂控制接口，负责协调用户交互与任务规划。请严格遵守以下规则：
## 主要职责
1. 日常对话：处理用户的一般性询问和指令（不能生成[CodEGEN]标记）
2. 需求分析：
当检测到操作需求时，你首先判断图片中有什么物体，然后根据需求执行以下流程：
    任务规划：
        可执行时，生成符合指令的任务规划，并在前面加上**[CodEGEN] 标记**，交给Coder Agent去生成代码。需要继续和用户交互则不生成**[CodEGEN] 标记**，
        不可执行时，不生成**[CodEGEN] 标记**，而是请求用户澄清或提供额外信息。
    澄清策略：
        模糊指令推测：对于部分可推测目标的模糊指令，可结合场景理解和任务需求进行规划。
        基于选项的澄清：对于可能存在多种解释的指令，生成多个可选任务规划方案，并请用户选择最符合意图的方案。
        主动引导式澄清：当用户的指令过于宽泛或不完整时，可主动提问，引导用户提供更具体的信息。
    物体选择策略（优化点）
        指令涉及大类时：  
        先识别场景中所有的属于该类的物体。  
        根据你的理解判断是请用户选择具体的目标物体还是直接执行任务规划。  
    场景与物体分析：
        结合摄像机观测到的物体与指令内容，提供多个可能的名称，增强任务执行的成功率。
    指令涉及方位或距离时：  
        若不明确，则询问具体要求。
    物理可行性判断：
        在确认物理可行的前提下，才生成**[CodEGEN] 任务规划**。
        若指令无法执行（例如目标超出可操作范围、物理冲突等），应直接提示问题，而不是生成无效规划。
## 规划输出格式,你只能输出以下规划action
[CodEGEN]
{
    "steps": [
        {"action": "抓起", "target": "<物体名（英文。可提供多个替代名称）>"},
        {"action": "移动到目标位置上方5cm", "target": "<物体名（英文。可提供多个替代名称）>"},
        {"action": "以当前位置进行位置调整", "direction": "<只能是”前后左右上下“六种>", "distance": <距离，单位是米>},
        {"action": "松开夹爪"},
        {"action": "复位"}
    ]
}
示例：
1.把杯子向左移动1cm
[CodEGEN]
{
    "steps": [
        {"action": "抓起", "target": "cup"},
        {"action": "以当前位置进行位置调整", "direction": "left", "distance": 0.01},
        {"action": "松开夹爪"},
        {"action": "复位"}
    ]
}
2.把刀放到盘子上
[CodEGEN]
{
    "steps": [
        {"action": "抓起", "target": "knife..."},
        {"action": "移动到目标位置上方5cm", "target": "plate..."},
        {"action": "松开夹爪"},
        {"action": "复位"}
    ]
}
3. 把盒子放到碗旁边
[CodEGEN]
{
    "steps": [
        {"action": "抓起", "target": "box..."},
        {"action": "移动到目标位置上方5cm", "target": "bowl..."},
        {"action": "以当前位置进行位置调整", "direction": "right/left/front/back", "distance": 0.1},
        {"action": "松开夹爪"},
        {"action": "复位"}
    ]
}

注意：
- target 必须是你在图像中检测到的物体名或放置位名。
- 若想执行放到目标物体旁边等操作，可以用先移动到目标物位置再进行位置调整
- 不允许在任务规划中出现 “接着”，“然后”，“首先”等自然语言
请注意你不能生成代码。
""")

# 代码生成模型的系统提示（优化后）
CODE_SYSTEM_PROMPT = SystemMessage("""## 代码生成专家
你接收结构化的steps，使用你可用的API编写的Python代码，仅生成一个plan函数 def plan()：
## 代码规范
1. 严格使用提供的API接口
2. 必须包含异常处理
3. 添加注释
4. 写判断条件时注意变量的类型

## 可用API及其使用规范
1. 物体检测与抓取
    get_graspPosition()
        功能：获取场景中所有可行的抓取姿态（poses）。
        返回：所有可行的抓取姿势的列表。
    filter_grasp_by_detections(text, poses)
        功能：执行目标检测，然后筛选出适用于目标物体的抓取姿势。
        参数：
            text：目标物体名称（英文）。
            poses：get_graspPosition() 的返回结果，即所有可行的抓取姿态。
        返回：筛选后的抓取姿势列表 final_pose，其中 final_pose[0] 是最佳抓取姿态。若返回为空需重新调用get_graspPosition()再进行filter
    move_and_grasp(final_pose[0])
        功能：机械臂移动到目标位置并执行抓取动作。无需move_to移动
        参数：
            final_pose[0]：filter_grasp_by_detections() 返回的抓取姿势列表中的第一个，即最优抓取姿势。

2. 物体放置
    get_put_positon(text: str)
        功能：获取放置位置。
        参数：
            text（字符串）：目标放置位置的名称，例如 "table"。
        返回：放置位置上方的 坐标 target_location。
        若需放置到该位置旁边，需要结合move函数使用

3. 机械臂移动
    move(direction: str, distance: float = 0.1)
        功能：控制机械臂以当前位置为原点向指定方向移动指定距离。
        参数：
            direction（字符串）：移动方向，仅支持以下值：
                "up"（向上）
                "down"（向下）
                "left"（向左）
                "right"（向右）
                "front"（向前）
                "back"（向后）
                方向以 机械臂视角 为参考系。
            distance（浮点数）：移动的距离，单位为 米（m），范围 0.01m ~ 1.0m，默认值 0.1m。
        行为：机械臂沿指定方向移动指定距离。
    move_to(target_location) 
        移动到指定位置上方5cm
4. 其他操作
    ungrasp()
        功能：松开夹爪，释放物体。
    execute_reset()
        功能：机械臂复位，回到初始位置。
注意事项：
以上API直接使用即可，不用添加前缀。
若filter_grasp_by_detections(text, poses)返回结果为空，需要重新检测抓取姿态,最多重新检测三次。。
请注意你只能生成plan函数，不能输出其他内容。
你在接收到Checker Agent的反馈后需要根据反馈完善代码。

以下是格式示例：
```python
def plan():
    ...
```
""")
# 代码验证系统提示
VALIDATION_SYSTEM_PROMPT = SystemMessage("""## 代码安全审计专家
请严格检查以下Python代码是否符合安全规范：

### 审计标准
1. **API使用规范**
   - 只能使用以下API(已经定义好了不需要重新实现）：
   get_graspPosition, 
   detect_by_text, 
   filter_grasp_by_detections, 
   move_and_grasp，
   get_put_positon，
   execute_reset,
   move(direction: str, distance: float = 0.1)：
   参数：
            direction（字符串）：移动方向，仅支持以下值：
                "up"（向上）
                "down"（向下）
                "left"（向左）
                "right"（向右）
                "front"（向前）
                "back"（向后）
                方向以 机械臂视角 为参考系。
            distance（浮点数）：移动的距离，单位为 米（m），范围 0.01m ~ 1.0m，默认值 0.1m。
   ungrasp,
   move_to 移动到目标位置上方5cm
   - 禁止使用其他文件/网络操作

2. **异常处理**
   - 必须包含try-except块

3. **类型安全**
   - 输入参数若是字符串，则是英文

4. **流程合理**
   - 代码符合任务规划结果且不能出现逻辑错误


### 响应格式
验证通过：
✅ 代码符合安全规范

需要修改：                                       
❌ [问题描述]
[修改建议]
并且添加[需要修改]标记  
注意你不能输出代码
""")

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
os.environ["OPENAI_API_KEY"] = ""

API_KEY = "16632895-9544-4e0f-97b3-03755a73eca3"
Model_list = {"Doubao-vision-pro-32k": "Your API Keys",
              "Doubao-1.5-vision-pro-32k": "",
              "Doubao-pro-32K": "",
              "Doubao-1.5-pro-32k": ""
              }
# 初始化 OpenAI 语言模型
chat_model = ChatOpenAI(
    model=Model_list["Doubao-1.5-vision-pro-32k"],
    api_key=API_KEY,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

code_model = ChatOpenAI(
    model=Model_list["Doubao-pro-32K"],
    api_key=API_KEY,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

valid_model = ChatOpenAI(
    model=Model_list["Doubao-pro-32K"],
    api_key=API_KEY,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)


# 定义增强的状态结构
class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    code_history: Annotated[list, add_messages]
    valid_history: Annotated[list, add_messages]
    code_request: Annotated[str, "当前代码需求"]  # 存储任务描述
    validation_result: Annotated[str, "验证结果"]
    validation_feedback: Annotated[list, add_messages]

# 创建工作流
workflow = StateGraph(state_schema=EnhancedState)


# _ = create_initial_state()
# 对话处理节点
def call_model(state: EnhancedState):
    print("\033[96mPlanner Agent\033[0m")
    if len(state["messages"]) == 1:
        state["messages"].insert(0, MAIN_SYSTEM_PROMPT)

    response = chat_model.invoke(state["messages"])
    state["messages"].append(response)
    print("\033[94m" + state["messages"][-1].content + "\033[0m")  # 蓝色输出规划结果
    return state


Code_history = [CODE_SYSTEM_PROMPT]


# 代码生成节点
def generate_code(state: EnhancedState):
    print("\033[92mCoder Agent\033[0m")  # 绿色
    if len(state["code_history"]) == 0:
        state["code_history"].insert(0, CODE_SYSTEM_PROMPT)
    # 提取代码生成需求
    last_msg = state["messages"][-1].content
    code_request = last_msg.split("[CodEGEN]")[-1].strip()
    state["code_request"] = code_request
    # **关键改动**：如果有 Checker 反馈，则合并到 code_request
    if state.get("validation_feedback"):
        feedback_text = "\n".join([msg.content for msg in state["validation_feedback"]])
        code_request += f"\n（请根据以下反馈优化代码：\n{feedback_text}\n）"

    state["code_history"].append(HumanMessage(content=code_request))
    # 生成代码
    code_response = code_model.invoke(state["code_history"])
    state["messages"].append(code_response)
    state["code_history"].append(code_response)
    print("\033[92m" + state["code_history"][-1].content + "\033[0m")  # 绿色输出生成的代码
    return state


Valid_history = [VALIDATION_SYSTEM_PROMPT]


# 代码验证节点
def valid_code(state: EnhancedState):
    print("\033[93mChecker Agent\033[0m")  # 黄色
    if len(state["valid_history"]) == 0:
        state["valid_history"].insert(0, VALIDATION_SYSTEM_PROMPT)
    # 提取规划和代码
    # 获取最后两条消息（需求+代码）
    context = "\n".join([
        f"需求：{state['code_request']}",
        f"代码：{state['code_history'][-1].content}"
    ])
    state["valid_history"].append(HumanMessage(content=context))
    valid_response = code_model.invoke(state["valid_history"])
    state["valid_history"].append(valid_response)
    state["validation_result"] = valid_response.content
    state["validation_feedback"].append(valid_response)
    print("\033[93m" + state["valid_history"][-1].content + "\033[0m")  # 黄色输出验证反馈
    # Valid_history.append(AIMessage(content=valid_response))
    return state


# 路由逻辑
# def router(state: EnhancedState):
#     last_msg = state["messages"][-1].content
#     return "code_generator" if "[CodEGEN]" in last_msg else END
def router(state: EnhancedState):
    last_msg = state["messages"][-1].content

    if "[CodEGEN]" in last_msg:
        return "code_generator"


    elif state.get("validation_result") == "验证通过":
        return END
    elif "需要修改" in state.get("validation_result", ""):
        return "code_generator"  # 返回代码生成器修改
    else:
        return END  # 继续验证


# 构建工作流
workflow.add_node("model", call_model)
workflow.add_node("code_generator", generate_code)
workflow.add_node("validator", valid_code)

workflow.set_entry_point("model")

# 配置条件转移
workflow.add_conditional_edges(
    "model",
    router,
    {"code_generator": "code_generator", END: END}
)

workflow.add_conditional_edges(
    "code_generator",
    lambda _: "validator",  # 代码生成后必须验证
    {"validator": "validator"}
)

workflow.add_conditional_edges(
    "validator",
    router,
    {"code_generator": "code_generator", END: END}
)

# 初始化应用
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 交互式多轮对话
config = {"configurable": {"thread_id": "chat_session_1"}}
messages = []


def get_input():
    task = input("请输入指令:").strip()
    # 支持多图上传逻辑
    image_path = "1.jpg"
    # 将图片转为Base64编码
    base64_image = encode_image(image_path)
    return task, base64_image


# 增强版图像编码（支持多图）
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


print("开始对话（输入 'exit' 退出）")


while True:
    # user_input = input("你: ")
    # if user_input.lower() == "exit":
    #     print("对话结束")
    #     break
    user_input, images = get_input()
    messages = [HumanMessage(
        content=[
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{images}"}, },
        ],
    )]
    output = app.invoke({"messages": messages}, config)  # 调用工作流

    ai_response = output["messages"][-1].content  # 获取 AI 回复
    print("AI:", ai_response)

    # 检查 AI 回复是否包含 Python 代码
    pattern = r"```python\n([\s\S]*?)```"
    match = re.search(pattern, ai_response, re.DOTALL)

    if match:
        # 如果 AI 回复中有代码块，提取并执行
        em_grasp.videocam.start_recording()
        code_block = match.group(1)
        # print(f"执行的 Python 代码块:\n{code_block}")
        try:
            exec(code_block + "\nplan()")  # 执行代码块
        except Exception as e:
            print(f"代码执行错误: {e}")
        em_grasp.videocam.stop_recording(save_to_filename='1.mp4', fps=60)
        em_grasp.scene.reset() #重置环境
    else:
        # 如果没有代码块，继续对话
        print("继续对话...")

