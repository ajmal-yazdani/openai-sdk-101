import os

import chainlit as cl
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import find_dotenv, load_dotenv
from openai import AsyncAzureOpenAI

# Load the environment variables from the .env file
load_dotenv(find_dotenv())

azure_openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
)

# Initialize the OpenAIChatCompletionsModel with the Azure OpenAI client
model = OpenAIChatCompletionsModel(model="gpt-4o", openai_client=azure_openai_client)

# Set up the RunConfig with the model and Azure OpenAI client
config = RunConfig(model=model, tracing_disabled=True)

# Create the agent with the model and instructions
agent: Agent = Agent(
    name="Assistant", instructions="You are a helpful assistant", model=model
)


@cl.on_chat_start
async def handle_start_chat():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! How can I assist you today?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    # Get history or initialize if it doesn't exist
    history = cl.user_session.get("history")
    if history is None:
        history = []
        cl.user_session.set("history", history)

    print(f"History: {history}")
    # Append the new message to the history
    history.append(
        {
            "role": "user",
            "content": message.content,
        }
    )

    # This function will be called when a message is sent in the Chainlit app
    result = await Runner.run(agent, input=history, run_config=config)

    history.append(
        {
            "role": "assistant",
            "content": result.final_output,
        }
    )

    # Make sure to update the session with the new history
    cl.user_session.set("history", history)

    # Send a fake response back to the user
    await cl.Message(content=result.final_output).send()
