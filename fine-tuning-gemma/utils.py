def preprocess(sample, tokenizer):
      messages = sample["messages"]
      first_message = messages[0]

      # Instead of adding a system message, we merge the content into the first user message
      if first_message["role"] == "system":
          system_message_content = first_message["content"]
          # Merge system content with the first user message
          messages[1]["content"] = system_message_content + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n" + messages[1]["content"]
          # Remove the system message from the conversation
          messages.pop(0)

      return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

