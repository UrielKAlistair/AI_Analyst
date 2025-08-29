from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import io
import contextlib

load_dotenv()

# Use a pre-built tool for searching
search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="google_search",
    description="A tool for performing a web search to find current information. Input should be a specific search query.",
    func=search.run,
)


@tool
def code_interpreter(code: str) -> str:
    """Executes a Python script in a restricted environment. Input should be a valid Python script."""
    try:
        # Restrict environment (no imports except built-ins allowed by default)
        safe_globals = {"__builtins__": __builtins__}
        safe_locals = {}

        # Capture stdout
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, safe_globals, safe_locals)

        return buffer.getvalue().strip() or "Execution finished with no output."
    except Exception as e:
        return f"Execution Error: {e}"


# Put all tools in a list
tools = [search_tool, code_interpreter]


class Worker:
    def __init__(self) -> None:

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # A popular agent type
            verbose=True,  # Set to True to see the thought process
        )

    def __call__(self, task_prompt: str):
        # The agent executor handles everything:
        # 1. It looks at the task_prompt
        # 2. Decides which tool to call (or to not call one)
        # 3. Executes the tool
        # 4. Takes the tool's output and uses it to generate a final response
        try:
            result = self.agent_executor.invoke({"input": task_prompt})
            return result["output"]
        except Exception as e:
            # Handle potential errors from the agent execution
            return f"An error occurred during task execution: {e}"
