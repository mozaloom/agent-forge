import os
import base64
from dotenv import load_dotenv
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, LiteLLMModel


# Load environment variables from .env file
load_dotenv()

LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST") + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

HF_TOKEN = os.getenv("HF_TOKEN")

# Create a TracerProvider for OpenTelemetry
trace_provider = TracerProvider()

# Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

# Set the global default tracer provider
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Instrument smolagents with the configured provider
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
'''
# Create a simple agent to test instrumentation
agent = CodeAgent(
    tools=[],
    model=HfApiModel()
)

agent.run("1+1=")
'''


model=LiteLLMModel(model_id="gemini/gemini-2.0-flash-lite", api_key=os.environ.get('GOOGLE_API_KEY'))

search_tool = DuckDuckGoSearchTool()
#agent = CodeAgent(tools=[search_tool], model=HfApiModel())
agent = CodeAgent(tools=[search_tool], model=model)


print(agent.run("How many Rubik's Cubes could you fit inside the Notre Dame Cathedral?"))
'''
with tracer.start_as_current_span("Smolagent-Trace") as span:
    span.set_attribute("langfuse.user.id", "smolagent-user-123")
    span.set_attribute("langfuse.session.id", "smolagent-session-123456789")
    span.set_attribute("langfuse.tags", ["city-question", "testing-agents"])

    agent.run("What is the capital of Germany?")

'''