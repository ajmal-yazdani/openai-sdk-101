import os
from typing import List, cast

import chainlit as cl
from agents import Agent, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
from agents.tool import function_tool
from dotenv import find_dotenv, load_dotenv
from openai import AsyncAzureOpenAI

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Check if any of the required environment variables are missing
required_vars = {
    "AZURE_OPENAI_API_KEY": azure_openai_api_key,
    "AZURE_OPENAI_API_VERSION": azure_openai_api_version,
    "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
    "AZURE_OPENAI_DEPLOYMENT": azure_openai_deployment,
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(
        f"Required environment variables are missing: {', '.join(missing_vars)}. "
        "Please ensure they are defined in your .env file."
    )


@cl.set_starters  # type: ignore
async def set_starters() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
        cl.Starter(
            label="Weather",
            message="Find the weather in Hyderabad.",
        ),
    ]


@function_tool
@cl.step(type="tool")
def get_weather(location: str, unit: str = "C") -> str:
    """
    Fetch the weather for a given location, returning a short description.
    """
    # Example logic
    return f"The weather in {location} is 22 degrees {unit}."


# Check if the required environment variables are set
@cl.on_chat_start
async def handle_start():
    # Set up the Azure OpenAI client
    # This is the async version of the Azure OpenAI client
    azure_openai_client = AsyncAzureOpenAI(
        api_key=cast(str, azure_openai_api_key),
        api_version=cast(str, azure_openai_api_version),
        azure_endpoint=cast(str, azure_openai_endpoint),
        azure_deployment=cast(str, azure_openai_deployment),
    )

    # Initialize the OpenAIChatCompletionsModel with the Azure OpenAI client
    model = OpenAIChatCompletionsModel(
        model="gpt-4o", openai_client=azure_openai_client
    )

    # Set up the RunConfig with the model and Azure OpenAI client
    config = RunConfig(model=model, tracing_disabled=True)

    # Initialize an empty chat history in the session.
    cl.user_session.set("chat_history", [])

    # Create the agent with the model and instructions
    # set the config in the session
    cl.user_session.set("config", config)
    agent: Agent = Agent(
        name="Assistant", instructions="You are a helpful assistant", model=model
    )
    agent.tools.append(get_weather)
    # set the agent in the session
    cl.user_session.set("agent", agent)

    await cl.Message(content="Hello! How can I assist you today?").send()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []

    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")

        result = Runner.run_sync(agent, history, run_config=config)
        response_content = result.final_output

        # Update the thinking message with the actual response
        msg.content = response_content
        await msg.update()

        # Append the assistant's response to the history.
        history.append({"role": "developer", "content": response_content})
        # NOTE: Here we are appending the response to the history as a developer message.
        # This is a BUG in the agents library.
        # The expected behavior is to append the response to the history as an assistant message.

        # Update the session with the new history.
        cl.user_session.set("chat_history", history)

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")
    except Exception as e:
        # Fix: First set the content property, then call update() without parameters
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
