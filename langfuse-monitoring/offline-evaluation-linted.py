"""Offline evaluation script for Langfuse monitoring with OpenTelemetry."""
import base64
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import format_trace_id
from smolagents import CodeAgent, LiteLLMModel
from openinference.instrumentation.smolagents import SmolagentsInstrumentor


def main():
    """Run the offline evaluation end-to-end."""
    load_dotenv()

    # Validate required env vars
    lf_pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_sec = os.getenv("LANGFUSE_SECRET_KEY")
    lf_host = os.getenv("LANGFUSE_HOST")
    hf_token = os.getenv("HF_TOKEN")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not all([lf_pub, lf_sec, lf_host]):
        sys.exit("ERROR: Missing one of LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY or LANGFUSE_HOST")

    # Configure OTLP exporter
    auth = f"{lf_pub}:{lf_sec}"
    b64_auth = base64.b64encode(auth.encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{lf_host}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {b64_auth}"

    # Setup OpenTelemetry tracing
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)

    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

    # Load GSM8K benchmark
    ds = load_dataset("openai/gsm8k", "main", split="train")
    df = pd.DataFrame(ds)  # type: ignore

    # Initialize Langfuse
    lf = Langfuse()
    dataset_name = "gsm8k_dataset_huggingface"
    lf.create_dataset(
        name=dataset_name,
        description="GSM8K benchmark dataset from Hugging Face",
        metadata={"date": "2025-03-10", "type": "benchmark"},
    )

    # Upload first 10 items
    for idx, row in df.head(10).iterrows():
        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={"text": row["question"]},
            expected_output={"text": row["answer"]},
            metadata={"source_index": idx},
        )

    # Prepare the agent
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash-lite",
        api_key=google_api_key,
    )
    agent = CodeAgent(tools=[], model=model, add_base_tools=True)

    def run_smolagent(question: str):
        """Run the SmolAgent with tracing and record to Langfuse."""
        with tracer.start_as_current_span("smolagent_trace") as span:
            span.set_attribute("langfuse.tag", "dataset-run")
            output = agent.run(question)

            span_context = trace.get_current_span().get_span_context()
            formatted_id = format_trace_id(span_context.trace_id)

            lf_trace = lf.trace(id=formatted_id, input=question, output=output)
        return lf_trace

    # Execute agent on dataset items
    dataset = lf.get_dataset(dataset_name)
    for item in dataset.items:
        lf_trace = run_smolagent(item.input["text"])
        item.link(
            lf_trace,
            run_name="smolagent-notebook-run-01",
            run_metadata={"model": model.model_id},
        )
        lf_trace.score(name="example_eval", value=1, comment="Demo evaluation")

    lf.flush()


if __name__ == "__main__":
    main()