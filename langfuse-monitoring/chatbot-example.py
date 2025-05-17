import os
import base64
import pandas as pd
import gradio as gr
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


trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

langfuse = Langfuse()
model= LiteLLMModel(model_id="gemini/gemini-2.0-flash-lite", api_key=os.environ.get('GOOGLE_API_KEY'))
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

formatted_trace_id = None  # We'll store the current trace_id globally for demonstration

def respond(prompt, history):
    with trace.get_tracer(__name__).start_as_current_span("Smolagent-Trace") as span:
        output = agent.run(prompt)

        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
        trace_id = span_context.trace_id
        global formatted_trace_id
        formatted_trace_id = str(format_trace_id(trace_id))
        langfuse.trace(id=formatted_trace_id, input=prompt, output=output)

    history.append({"role": "assistant", "content": str(output)})
    return history

def handle_like(data: gr.LikeData):
    # For demonstration, we map user feedback to a 1 (like) or 0 (dislike)
    if data.liked:
        langfuse.score(
            value=1,
            name="user-feedback",
            trace_id=formatted_trace_id
        )
    else:
        langfuse.score(
            value=0,
            name="user-feedback",
            trace_id=formatted_trace_id
        )

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat", type="messages")
    prompt_box = gr.Textbox(placeholder="Type your message...", label="Your message")

    # When the user presses 'Enter' on the prompt, we run 'respond'
    prompt_box.submit(
        fn=respond,
        inputs=[prompt_box, chatbot],
        outputs=chatbot
    )

    # When the user clicks a 'like' button on a message, we run 'handle_like'
    chatbot.like(handle_like, None, None)

demo.launch()
