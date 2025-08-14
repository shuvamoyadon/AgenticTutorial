# 🔗 Connecting Agentic Patterns to Tools

## 📋 Overview

This guide demonstrates how an AI agent can use the "tool use" agentic pattern to interact with external tools (like APIs or web searches). It covers the essential building blocks for creating effective AI agents and provides practical Python implementations using the Groq API.

---

## 🛠️ Basic Tool Use Pattern Implementation

### Python Implementation with OpenAI Function Calling

```python
import openai

openai.api_key = 'YOUR_API_KEY'

# Define tool functions available to the agent
def calculator_tool(expression):
    try:
        result = eval(expression)
        return f"Calculator result: {result}"
    except Exception as e:
        return f"Error: {e}"

def weather_api_tool(location):
    fake_weather = {"New York": "Sunny, 25°C", "Mumbai": "Cloudy, 30°C"}
    return f"Weather in {location}: {fake_weather.get(location, 'No data')}"

# Registry for tool implementations
tools = {
    "calculator": calculator_tool,
    "weather": weather_api_tool,
}

# Define the schema for function calling
functions = [
    {
        "name": "calculator",
        "description": "A basic calculator",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Expression to calculate, like '2 + 2'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "weather",
        "description": "Get weather info for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Mumbai'"}
            },
            "required": ["location"]
        }
    }
]

def agent_respond_with_llm(user_query):
    # Step 1: Ask the LLM to choose a tool via function calling
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use gpt-4-turbo or latest function-calling capable model
        messages=[{"role": "user", "content": user_query}],
        functions=functions,
        function_call="auto",
    )
    choice = response.choices[0]
    if choice.finish_reason == "function_call":
        # LLM decided to use a tool; extract tool name and arguments
        name = choice.message.function_call.name
        args = eval(choice.message.function_call.arguments)
        tool_response = tools[name](**args)
        return tool_response
    else:
        # LLM chose to just respond naturally
        return choice.message.content

# Example test cases:
print(agent_respond_with_llm("Calculate 7 * 8"))
print(agent_respond_with_llm("What's the weather in Mumbai?"))
print(agent_respond_with_llm("Tell me a joke."))  # Handled directly by the LLM
```

---

## 🏗️ Essential Building Blocks for Effective AI Agents

| Building Block | Description |
|:--|:--|
| **🎯 Goal Definition** | Clear specification of the agent's primary objective or task. |
| **👁️ Perception/Input** | Mechanisms for receiving and interpreting external data (e.g., user queries, sensor data, APIs). |
| **🧠 Reasoning & Planning** | Logic or models (often LLMs) for understanding context, decomposing problems, and selecting steps |
| **🔧 Tool Use/Action Modules** | Libraries or APIs that enable the agent to interact with external systems or perform computations. |
| **💾 Memory/State Management** | Components for storing context, conversation history, or relevant world/model knowledge. |
| **⚖️ Decision-Making Policy** | Framework or algorithms that help the agent choose when/which action or tool to invoke. |
| **📈 Learning & Adaptation** | Ability to improve through feedback, experience, or user corrections (e.g., RL, fine-tuning). |
| **💬 Interface/Communication** | Systems for engaging with users or other agents (e.g., chat UIs, voice, API endpoints). |
| **🛡️ Safety & Alignment** | Constraints and safeguards to ensure agent acts within ethical, legal, and intended boundaries. |
| **🔌 Modularity & Extensibility** | Design that allows easy integration of new tools, skills, and capabilities. |

---

## 🚀 Python Implementation Using Groq API

### 1. 🎯 Goal Definition
```python
# Agent's main objective, defined for prompt clarity and reasoning
AGENT_GOAL = "Summarize scientific articles so non-experts can understand them."
```

### 2. 👁️ Perception/Input
```python
# Example: Receive input from user or an external API
user_query = input("Ask the agent: ")
```

### 3. 🧠 Reasoning & Planning (LLM via Groq)
```python
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Standard LLM call for reasoning/planning, providing system prompt for goal/context
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": f"You are an agent. Goal: {AGENT_GOAL}"},
        {"role": "user", "content": user_query}
    ],
    model="llama3-8b-8192"
)
print("Agent:", response.choices[0].message.content)
```

### 4. 🔧 Tool Use / Action Modules
```python
# Simple tool functions, agent decides (can also be triggered by the LLM's suggestion)
def search_tool(query):
    # Placeholder: In production, call a search API
    return f"Search results for '{query}': [stub]"

def math_tool(expression):
    try:
        return f"Result: {eval(expression)}"
    except:
        return "Invalid math expression."

# Example usage:
# result = search_tool("Groq API advantages")
# print(result)
```

### 5. 💾 Memory / State Management
```python
# Maintain chat/conversation history for continuity and context
conversation_history = [
    {"role": "system", "content": f"Agent goal: {AGENT_GOAL}"}
]
conversation_history.append({"role": "user", "content": user_query})
conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

# For context persistence, pass conversation_history to future LLM calls
```

### 6. ⚖️ Decision-Making Policy
```python
# Decide next action/tool based on analysis (simple rule-based, LLM-powered, or both)
def choose_action(query):
    if "search" in query.lower():
        return search_tool(query)
    elif any(char.isdigit() for char in query) and any(op in query for op in "+-*/"):
        return math_tool(query)
    else:
        # Default to LLM
        return response.choices[0].message.content
```

### 7. 📈 Learning & Adaptation
```python
# Collect user feedback for incremental improvement/future tuning
feedback = input("Was this answer helpful? (y/n): ")
log_entry = {
    "query": user_query,
    "response": response.choices[0].message.content,
    "feedback": feedback
}
# Store log_entry for analysis or model fine-tuning downstream
```

### 8. 🛡️ Safety & Alignment
```python
# Filter/block unsafe requests before dispatching to the LLM
unsafe_keywords = ["hack", "illegal", "scam"]
if any(word in user_query.lower() for word in unsafe_keywords):
    print("Request blocked for safety reasons.")
else:
    # Proceed to agent reasoning, tool use, etc.
    pass
```

### 9. 🔌 Modularity & Extensibility
```python
# Easily add new tools or swap components with a Python dictionary registry
TOOLS = {
    "search": search_tool,
    "math": math_tool,
    # Add more as needed...
}

# Example: Dynamically call a tool
tool_to_use = "search"
if tool_to_use in TOOLS:
    output = TOOLS[tool_to_use]("Groq API")
    print(output)
```

---

## 🎯 Key Takeaways

- **🔗 Tool Integration**: Effective AI agents seamlessly integrate external tools through well-defined interfaces
- **🏗️ Modular Architecture**: Building blocks approach allows for flexible and extensible agent design
- **🧠 Smart Decision Making**: Agents should intelligently choose between direct LLM responses and tool usage
- **💾 Context Management**: Maintaining conversation history and state is crucial for coherent interactions
- **🛡️ Safety First**: Always implement safety measures and input validation
- **📈 Continuous Learning**: Collect feedback for ongoing improvement and adaptation

---

## 🚀 Next Steps

1. **Implement Real Tools**: Replace placeholder functions with actual API integrations
2. **Add Error Handling**: Implement robust error handling for tool failures
3. **Enhance Memory**: Add persistent storage for long-term memory
4. **Scale Architecture**: Design for multi-agent systems and complex workflows
5. **Monitor Performance**: Add logging and metrics for agent performance tracking

---

*This guide provides a foundation for building sophisticated AI agents that can effectively use tools and interact with their environment.*