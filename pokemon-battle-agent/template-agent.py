import os
import json
import asyncio
import random

# --- OpenAI ---
from openai import AsyncOpenAI, APIError

# --- Google Gemini ---
from google import genai
from google.genai import types

# --- Mistral AI ---
from mistralai import Mistral

# --- Poke-Env ---
from poke_env.player import Player
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from typing import Optional, Dict, Any, Union



class TemplateAgent(LLMAgentBase):
    """Uses Template AI API for decisions."""
    def __init__(self, api_key: str = None, model: str = "model-name", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.template_client = TemplateModelProvider(api_key=...)
        self.template_tools = list(self.standard_tools.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """Sends state to the LLM and gets back the function call decision."""
        system_prompt = (
            "You are a ..."
        )
        user_prompt = f"..."

        try:
            response = await self.template_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = response.choices[0].message
            
            return {"decision": {"name": function_name, "arguments": arguments}}

        except Exception as e:
            print(f"Unexpected error during call: {e}")
            return {"error": f"Unexpected error: {e}"}