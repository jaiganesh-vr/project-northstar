import os
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-oss-120b"

def build_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set. Add it to your environment or a .env file.")
    return ChatGroq(model=MODEL, temperature=0.3, max_retries=2)

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

# LLM with bound tool
tools = [add, multiply, divide]
llm = build_llm()
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

# Compile graph
graph = builder.compile(checkpointer=memory)
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    print("Ask an arithmetic question (type 'exit' to quit)")
    seen = 0
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        messages = graph.invoke({"messages": [("human", question)]}, config)["messages"]
        # Memory returns the whole conversation, so only print the new messages
        for m in messages[seen:]:
            m.pretty_print()
        seen = len(messages)
