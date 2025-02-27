from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

import os
os.environ['OPENAI_API_KEY'] = "sk-"
os.environ['TAVILY_API_KEY'] = "tvly-"

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from typing import Dict, List, Tuple
from langchain.agents import tool, initialize_agent, AgentType
import time
from langchain.load.dump import dumps
import json
from langchain.agents import tool



search = DuckDuckGoSearchRun()

tools = [search]

class Agent():
    base_url = "https://api.openai.com/v1"
    def __init__(self, defination, tools=[]):
        self.tools = tools
        self.defination = defination

    def run(self, instruction: str, max_turn=10):
        action_list = []

        self.llm = ChatOpenAI(model="deepseek-chat", 
                 openai_api_base="https://api.deepseek.com", 
                 openai_api_key="sk-", temperature=0.2,
                 ).bind_tools(self.tools)
        final_answer = ""
        while max_turn > 0:
            try:
                agent = initialize_agent(
                    tools=self.tools,
                    llm=self.llm,
                    verbose=True,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    return_intermediate_steps=True,
                    max_execution_time=120,  # seconds
                    max_iterations=6,  # 决定了最大的迭代次数
                )
                
                input_text = f"{self.defination}.\n{instruction}"
                
                response = agent({"input":input_text})
                action_list = []
                response = json.loads(dumps(response, pretty=True))
                for step in response["intermediate_steps"]:
                    print(step[1])
                    action_list.append({"action": step[0]["kwargs"], "feedback": step[1]})
                final_answer = response["output"]
                return final_answer
            
            except Exception as e:
                print(f"Error occurred: {e}")
                print("Retrying...")
                time.sleep(1)
                max_turn -= 1
        
        return final_answer
    

agent = Agent("你是爱丽丝，一个可爱的魔法师", tools=tools)
instruction = "你好，我想知道今天青岛的天气怎么样？"
response = agent.run(instruction)

