import os
import base64
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from langfuse import Langfuse
from opentelemetry.trace import format_trace_id
from smolagents import (CodeAgent, HfApiModel, LiteLLMModel)
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

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


# Fetch GSM8K from Hugging Face
dataset = load_dataset("openai/gsm8k", 'main', split='train')
df = pd.DataFrame(dataset)
#print("First few rows of GSM8K dataset:")
#print(df.head())

langfuse = Langfuse()

langfuse_dataset_name = "gsm8k_dataset_huggingface"

# Create a dataset in Langfuse
langfuse.create_dataset(
    name=langfuse_dataset_name,
    description="GSM8K benchmark dataset uploaded from Huggingface",
    metadata={
        "date": "2025-03-10", 
        "type": "benchmark"
    }
)

for idx, row in df.iterrows():
    langfuse.create_dataset_item(
        dataset_name=langfuse_dataset_name,
        input={"text": row["question"]},
        expected_output={"text": row["answer"]},
        metadata={"source_index": idx}
    )
    if idx >= 9: # Upload only the first 10 items for demonstration
        break


# running an agent on the dataset
model=LiteLLMModel(model_id="gemini/gemini-2.0-flash-lite", api_key=os.environ.get('GOOGLE_API_KEY'))

agent = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True
)

def run_smolagent(question):
    with tracer.start_as_current_span("Smolagent-Trace") as span:
        span.set_attribute("langfuse.tag", "dataset-run")
        output = agent.run(question)

        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
        trace_id = span_context.trace_id
        formatted_trace_id = format_trace_id(trace_id)

        langfuse_trace = langfuse.trace(
            id=formatted_trace_id, 
            input=question, 
            output=output
        )
    return langfuse_trace, output

dataset = langfuse.get_dataset(langfuse_dataset_name)

# Run our agent against each dataset item (limited to first 10 above)
for item in dataset.items:
    langfuse_trace, output = run_smolagent(item.input["text"])

    # Link the trace to the dataset item for analysis
    item.link(
        langfuse_trace,
        run_name="smolagent-notebook-run-01",
        run_metadata={ "model": model.model_id }
    )

    # Optionally, store a quick evaluation score for demonstration
    langfuse_trace.score(
        name="<example_eval>",
        value=1,
        comment="This is a comment"
    )

# Flush data to ensure all telemetry is sent
langfuse.flush()
