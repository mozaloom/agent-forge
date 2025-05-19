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

# --- Helper Function & Base Class ---
def normalize_name(name: str) -> str:
    """Lowercase and remove non-alphanumeric characters."""
    return "".join(filter(str.isalnum, name)).lower()

STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        "name": "choose_move",
        "description": "Selects and executes an available attacking or status move.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_name": {
                    "type": "string",
                    "description": "The exact name or ID (e.g., 'thunderbolt', 'swordsdance') of the move to use. Must be one of the available moves.",
                },
            },
            "required": ["move_name"],
        },
    },
    "choose_switch": {
        "name": "choose_switch",
        "description": "Selects an available Pokémon from the bench to switch into.",
        "parameters": {
            "type": "object",
            "properties": {
                "pokemon_name": {
                    "type": "string",
                    "description": "The exact name of the Pokémon species to switch to (e.g., 'Pikachu', 'Charizard'). Must be one of the available switches.",
                },
            },
            "required": ["pokemon_name"],
        },
    },
}

# --- OpenAI Tools Schema (with 'type' field) ---
OPENAI_TOOL_SCHEMA = {
    "choose_move": {
        "type": "function",
        "function": {
            "name": "choose_move",
            "description": "Selects and executes an available attacking or status move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "move_name": {
                        "type": "string",
                        "description": "The exact name or ID (e.g., 'thunderbolt', 'swordsdance') of the move to use. Must be one of the available moves.",
                    },
                },
                "required": ["move_name"],
            },
        }
    },
    "choose_switch": {
        "type": "function",
        "function": {
            "name": "choose_switch",
            "description": "Selects an available Pokémon from the bench to switch into.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pokemon_name": {
                        "type": "string",
                        "description": "The exact name of the Pokémon species to switch to (e.g., 'Pikachu', 'Charizard'). Must be one of the available switches.",
                    },
                },
                "required": ["pokemon_name"],
            },
        }
    },
}


class LLMAgentBase(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        self.battle_history = []

    def _format_battle_state(self, battle: Battle) -> str:
        active_pkmn = battle.active_pokemon
        active_pkmn_info = f"Your active Pokemon: {active_pkmn.species} " \
                           f"(Type: {'/'.join(map(str, active_pkmn.types))}) " \
                           f"HP: {active_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"Status: {active_pkmn.status.name if active_pkmn.status else 'None'} " \
                           f"Boosts: {active_pkmn.boosts}"

        opponent_pkmn = battle.opponent_active_pokemon
        opp_info_str = "Unknown"
        if opponent_pkmn:
            opp_info_str = f"{opponent_pkmn.species} " \
                           f"(Type: {'/'.join(map(str, opponent_pkmn.types))}) " \
                           f"HP: {opponent_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"Status: {opponent_pkmn.status.name if opponent_pkmn.status else 'None'} " \
                           f"Boosts: {opponent_pkmn.boosts}"
        opponent_pkmn_info = f"Opponent's active Pokemon: {opp_info_str}"

        available_moves_info = "Available moves:\n"
        if battle.available_moves:
            available_moves_info += "\n".join(
                [f"- {move.id} (Type: {move.type}, BP: {move.base_power}, Acc: {move.accuracy}, PP: {move.current_pp}/{move.max_pp}, Cat: {move.category.name})"
                 for move in battle.available_moves]
            )
        else:
             available_moves_info += "- None (Must switch or Struggle)"

        available_switches_info = "Available switches:\n"
        if battle.available_switches:
              available_switches_info += "\n".join(
                  [f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, Status: {pkmn.status.name if pkmn.status else 'None'})"
                   for pkmn in battle.available_switches]
              )
        else:
            available_switches_info += "- None"

        state_str = f"{active_pkmn_info}\n" \
                    f"{opponent_pkmn_info}\n\n" \
                    f"{available_moves_info}\n\n" \
                    f"{available_switches_info}\n\n" \
                    f"Weather: {battle.weather}\n" \
                    f"Terrains: {battle.fields}\n" \
                    f"Your Side Conditions: {battle.side_conditions}\n" \
                    f"Opponent Side Conditions: {battle.opponent_side_conditions}"
        return state_str.strip()

    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        normalized_name = normalize_name(move_name)
        # Prioritize exact ID match
        for move in battle.available_moves:
            if move.id == normalized_name:
                return move
        # Fallback: Check display name (less reliable)
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                print(f"Warning: Matched move by display name '{move.name}' instead of ID '{move.id}'. Input was '{move_name}'.")
                return move
        return None

    def _find_pokemon_by_name(self, battle: Battle, pokemon_name: str) -> Optional[Pokemon]:
        normalized_name = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            # Normalize the species name for comparison
            if normalize_name(pkmn.species) == normalized_name:
                return pkmn
        return None

    async def choose_move(self, battle: Battle) -> str:
        battle_state_str = self._format_battle_state(battle)
        decision_result = await self._get_llm_decision(battle_state_str)
        print(decision_result)
        decision = decision_result.get("decision")
        error_message = decision_result.get("error")
        action_taken = False
        fallback_reason = ""

        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        chat_msg = f"AI Decision: Using move '{chosen_move.id}'."
                        print(chat_msg)
                        return self.create_order(chosen_move)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid move '{move_name}'."
                else:
                     fallback_reason = "LLM 'choose_move' called without 'move_name'."
            elif function_name == "choose_switch":
                pokemon_name = args.get("pokemon_name")
                if pokemon_name:
                    chosen_switch = self._find_pokemon_by_name(battle, pokemon_name)
                    if chosen_switch and chosen_switch in battle.available_switches:
                        action_taken = True
                        chat_msg = f"AI Decision: Switching to '{chosen_switch.species}'."
                        print(chat_msg)
                        return self.create_order(chosen_switch)
                    else:
                        fallback_reason = f"LLM chose unavailable/invalid switch '{pokemon_name}'."
                else:
                    fallback_reason = "LLM 'choose_switch' called without 'pokemon_name'."
            else:
                fallback_reason = f"LLM called unknown function '{function_name}'."

        if not action_taken:
            if not fallback_reason:
                 if error_message:
                     fallback_reason = f"API Error: {error_message}"
                 elif decision is None:
                      fallback_reason = "LLM did not provide a valid function call."
                 else:
                      fallback_reason = "Unknown error processing LLM decision."

            print(f"Warning: {fallback_reason} Choosing random action.")

            if battle.available_moves or battle.available_switches:
                 return self.choose_random_move(battle)
            else:
                 print("AI Fallback: No moves or switches available. Using Struggle/Default.")
                 return self.choose_default_move(battle)

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _get_llm_decision")


# --- Google Gemini Agent ---
class GeminiAgent(LLMAgentBase):
    """Uses Google Gemini API for decisions."""
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-pro-preview-03-25", avatar: str = "steven", *args, **kwargs):
        # Set avatar before calling parent constructor
        kwargs['avatar'] = avatar
        kwargs['start_timer_on_battle_start'] = True
        super().__init__(*args, **kwargs)
        self.model_name = model
        used_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not used_api_key:
            raise ValueError("Google API key not provided or found in GOOGLE_API_KEY env var.")
        
        # Initialize Gemini client using the correct API
        self.genai_client = genai.Client(api_key=used_api_key)
        
        # Configure the tools for function calling
        self.function_declarations = list(self.standard_tools.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """Sends state to the Gemini API and gets back the function call decision."""
        prompt = (
            "Based on the current battle state, decide the best action: either use an available move or switch to an available Pokémon. "
            "Consider type matchups, HP, status conditions, field effects, entry hazards, and potential opponent actions. "
            "Only choose actions listed as available using their exact ID (for moves) or species name (for switches). "
            "Use the provided functions to indicate your choice.\n\n"
            f"Current Battle State:\n{battle_state}\n\n"
            "Choose the best action by calling the appropriate function ('choose_move' or 'choose_switch')."
        )

        try:
            # Configure tools using the Gemini API format
            tools = genai.types.Tool(function_declarations=self.function_declarations)
            config = genai.types.GenerateContentConfig(tools=[tools],automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            
            # Send request to the model
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            try:
                function_calls = response.function_calls
                if function_calls:
                    function_name = function_calls[0].name
                    arguments = function_calls[0].args
                    return {"decision": {"name": function_name, "arguments": arguments}}
                else:
                    return {"error": "Gemini did not return a function call."}
            except Exception as e:
                    return {"error": f"Model called unknown @function '{function_name}'."}
            
            # No function call found
            return {"error": "Gemini did not return a function call."}
            
        except Exception as e:
            print(f"Unexpected error during Gemini processing: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Unexpected error: {str(e)}"}


# --- OpenAI Agent ---
class OpenAIAgent(LLMAgentBase):
    """Uses OpenAI API for decisions."""
    def __init__(self, api_key: str = None, model: str = "gpt-4.1", avatar: str = "giovanni", *args, **kwargs):
        # Set avatar before calling parent constructor
        kwargs['avatar'] = avatar
        kwargs['start_timer_on_battle_start'] = True
        super().__init__(*args, **kwargs)
        self.model = model
        used_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not used_api_key:
            raise ValueError("OpenAI API key not provided or found in OPENAI_API_KEY env var.")
        self.openai_client = AsyncOpenAI(api_key=used_api_key)

        # Use the OpenAI-specific schema with type field
        self.openai_tools = list(OPENAI_TOOL_SCHEMA.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a skilled Pokemon battle AI. Your goal is to win the battle. "
            "Based on the current battle state, decide the best action: either use an available move or switch to an available Pokémon. "
            "Consider type matchups, HP, status conditions, field effects, entry hazards, and potential opponent actions. "
            "Only choose actions listed as available using their exact ID (for moves) or species name (for switches). "
            "Use the provided functions to indicate your choice."
        )
        user_prompt = f"Current Battle State:\n{battle_state}\n\nChoose the best action by calling the appropriate function ('choose_move' or 'choose_switch')."

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.openai_tools,
                tool_choice="auto",  # Let the model choose
                temperature=0.5,
            )
            message = response.choices[0].message
            print("OPENAI RESPONSE : ",response)
            # Check for tool calls in the response
            if message.tool_calls:
                tool_call = message.tool_calls[0]  # Get the first tool call
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or '{}')
                    if function_name in self.standard_tools:
                        return {"decision": {"name": function_name, "arguments": arguments}}
                    else:
                        return {"error": f"Model called unknown function '{function_name}'."}
                except json.JSONDecodeError:
                    return {"error": f"Error decoding function arguments: {tool_call.function.arguments}"}
            else:
                # Model decided not to call a function
                return {"error": f"OpenAI did not return a function call. Response: {message.content}"}

        except APIError as e:
            print(f"Error during OpenAI API call: {e}")
            return {"error": f"OpenAI API Error: {e.status_code} - {e.message}"}
        except Exception as e:
            print(f"Unexpected error during OpenAI API call: {e}")
            return {"error": f"Unexpected error: {e}"}


# --- Mistral Agent ---
class MistralAgent(LLMAgentBase):
    """Uses Mistral AI API for decisions."""
    def __init__(self, api_key: str = None, model: str = "mistral-large-latest", avatar: str = "alder", *args, **kwargs):
        # Set avatar before calling parent constructor
        kwargs['avatar'] = avatar
        kwargs['start_timer_on_battle_start'] = True
        super().__init__(*args, **kwargs)
        self.model = model
        used_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not used_api_key:
            raise ValueError("Mistral API key not provided or found in MISTRAL_API_KEY env var.")
        self.mistral_client = Mistral(api_key=used_api_key)

        # Convert standard schema to Mistral's tool format with "function" wrapper
        self.mistral_tools = []
        for tool_name, tool_schema in self.standard_tools.items():
            self.mistral_tools.append({
                "type": "function",
                "function": {
                    "name": tool_schema["name"],
                    "description": tool_schema["description"],
                    "parameters": tool_schema["parameters"]
                }
            })

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a skilled Pokemon battle AI. Your goal is to win the battle. "
            "Based on the current battle state, decide the best action: either use an available move or switch to an available Pokémon. "
            "Consider type matchups, HP, status conditions, field effects, entry hazards, and potential opponent actions. "
            "Only choose actions listed as available using their exact ID (for moves) or species name (for switches). "
            "Use the provided tools to indicate your choice."
        )
        user_prompt = f"Current Battle State:\n{battle_state}\n\nChoose the best action by calling the appropriate function ('choose_move' or 'choose_switch')."

        try:
            # Create the messages array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call the Mistral API with tool_choice set to "any" to force tool usage
            response = self.mistral_client.chat.complete(
                model=self.model,
                messages=messages,
                tools=self.mistral_tools,
                tool_choice="any",  # Force the model to use a tool
                temperature=0.3,
            )
            print("Mistral RESPONSE : ", response)
            
            # Check for tool calls in the response
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]  # Get the first tool call
                function_name = tool_call.function.name
                try:
                    # Parse the function arguments from JSON string
                    arguments = json.loads(tool_call.function.arguments or '{}')
                    if function_name in self.standard_tools:
                        return {"decision": {"name": function_name, "arguments": arguments}}
                    else:
                        return {"error": f"Model called unknown function '{function_name}'."}
                except json.JSONDecodeError:
                    return {"error": f"Error decoding function arguments: {tool_call.function.arguments}"}
            else:
                # Model did not return a tool call
                return {"error": f"Mistral did not return a tool call. Response: {message.content}"}

        except Exception as e:
            print(f"Error during Mistral API call: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Unexpected error: {str(e)}"}