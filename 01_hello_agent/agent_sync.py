import os

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

# Run the agent with the input text
result = Runner.run_sync(agent, "Hello, how are you.", run_config=config)
print(result)
# Print the agent's response
print(f"Agent Response: {result.final_output}")
