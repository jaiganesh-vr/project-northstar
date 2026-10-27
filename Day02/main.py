"""LangChain + Groq integration demo.

Setup:
    source langchain/bin/activate
    pip install -r requirements.txt
    cp .env.example .env      # then add your GROQ_API_KEY
    python main.py
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def build_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set. Add it to your environment or a .env file.")
    return ChatGroq(model=MODEL, temperature=0.3, max_retries=2)


def simple_invoke(llm: ChatGroq) -> None:
    """Call the model directly with a list of messages."""
    print("=== Simple invoke ===")
    response = llm.invoke(
        [
            ("system", "You are a concise assistant."),
            ("human", "What is Groq and why is it fast? Answer in two sentences."),
        ]
    )
    print(response.content, "\n")


def prompt_chain(llm: ChatGroq) -> None:
    """Compose prompt | llm | parser using LCEL."""
    print("=== Prompt chain ===")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert {domain} teacher. Explain things simply."),
            ("human", "Explain {topic} to a beginner in 3 bullet points."),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"domain": "AI", "topic": "LangChain"}), "\n")


def streaming(llm: ChatGroq) -> None:
    """Stream tokens as they are generated."""
    print("=== Streaming ===")
    for chunk in llm.stream("Write a haiku about fast inference."):
        print(chunk.content, end="", flush=True)
    print("\n")


def chat_loop(llm: ChatGroq) -> None:
    """Interactive chat that keeps conversation history."""
    print("=== Chat (type 'exit' to quit) ===")
    history = [("system", "You are a helpful assistant.")]
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        history.append(("human", user_input))
        print("Groq: ", end="", flush=True)
        reply = ""
        for chunk in llm.stream(history):
            print(chunk.content, end="", flush=True)
            reply += chunk.content
        print()
        history.append(("ai", reply))


def main() -> None:
    llm = build_llm()
    simple_invoke(llm)
    prompt_chain(llm)
    streaming(llm)
    chat_loop(llm)


if __name__ == "__main__":
    main()
