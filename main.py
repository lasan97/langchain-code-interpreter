from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv()

def python_agent():
    print("start...")

    instructions = """ You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, witch you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes
            that point to www.udemy.com, you have qrcode package installed already"""
        }
    )

def csv_agent():
    print("start...")

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # csv_agent.run("how many columns are there in file episode_info.csv")
    # csv_agent.run("which writer wrote the most episodes? how many episodes did he write?")
    # csv_agent.run("which writer wrote the least episodes? how many episodes did he write?")
    csv_agent.run("print seasions ascending order of the number of episodes they have")


if __name__ == '__main__':
    # python_agent()
    csv_agent()

