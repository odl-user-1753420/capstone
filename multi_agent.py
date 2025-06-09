import os
import asyncio
import re
import subprocess
from dotenv import load_dotenv
import webbrowser
import platform

# Load environment variables from .env file
load_dotenv()

# Fix for Python 3.10 on Windows asyncio event loop issue
if platform.system() == 'Windows':
    # Set the event loop policy to avoid the ProactorEventLoop issues
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Alternative fix for older versions
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion

# --- Initialize Kernel
kernel = Kernel()

# Use OpenAI directly (without Azure)
print("Using OpenAI service...")
kernel.add_service(
    AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
)

def load_system_message(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

# --- Load system prompts
ba_prompt = load_system_message("skills/BA/system_message.txt")
se_prompt = load_system_message("skills/SE/system_message.txt")
po_prompt = load_system_message("skills/PO/system_message.txt")

# --- Create ChatCompletionAgents
ba_agent = ChatCompletionAgent(
    name="BusinessAnalyst",
    description="Business Analyst persona for gathering and clarifying requirements.",
    kernel=kernel,
    instructions=ba_prompt
)
se_agent = ChatCompletionAgent(
    name="SoftwareEngineer",
    description="Software Engineer persona to implement requested features and produce HTML/JS code.",
    kernel=kernel,
    instructions=se_prompt
)
po_agent = ChatCompletionAgent(
    name="ProductOwner",
    description="Product Owner persona for reviewing and ensuring all requirements are met.",
    kernel=kernel,
    instructions=po_prompt
)

print("All ChatCompletionAgents created successfully:")
print(f"- {ba_agent.name}")
print(f"- {se_agent.name}")
print(f"- {po_agent.name}")

# --- Simple termination function instead of custom class to avoid Pydantic issues
async def should_terminate_conversation(history, max_iterations=15):
    """Simple function to check termination conditions."""
    
    # Count messages to estimate iterations
    iteration_count = len([msg for msg in history if isinstance(msg, ChatMessageContent)])
    
    if iteration_count >= max_iterations:
        print(f"⚠️ Maximum messages ({max_iterations}) reached. Auto-terminating to prevent errors.")
        return True
        
    # Check for user approval
    for msg in history:
        if (
            isinstance(msg, ChatMessageContent) and
            msg.role == AuthorRole.USER and
            "APPROVED" in msg.content.upper()
        ):
            print("✅ Termination condition met: User said APPROVED.")
            return True
            
    return False

# Create group chat without custom termination strategy to avoid Pydantic errors
group_chat = AgentGroupChat(
    agents=[ba_agent, se_agent, po_agent]
)
print("AgentGroupChat created and ready!")

# --- Callback to run after user says APPROVED
async def on_approved_callback():
    print("The user has APPROVED the work! Proceeding with final steps...")
    try:
        result = subprocess.run(
            ["./push_to_github.sh"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Git push successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error while pushing to GitHub:")
        print(e.stderr)

# --- Main agent system runner with proper cleanup
async def run_multi_agent(input_text: str):
    if not input_text.strip():
        print("Input text is empty. Please provide a valid prompt.")
        return

    try:
        # Add user message to kick off the conversation
        user_message = ChatMessageContent(
            role=AuthorRole.USER,
            content=input_text
        )
        await group_chat.add_chat_message(user_message)
        print("Added initial user message to chat history.")

        print("Streaming responses as they arrive...")
        
        # Add iteration counter for additional safety
        iteration_count = 0
        max_display_iterations = 15  # Reduced for better control
        
        try:
            async for content in group_chat.invoke():
                print(f"# {content.role}: '{content.content}'")
                iteration_count += 1
                
                # Safety check to prevent endless display loops
                if iteration_count >= max_display_iterations:
                    print(f"⚠️ Display limit ({max_display_iterations}) reached. Moving to final processing...")
                    break
                    
        except Exception as e:
            print(f"⚠️ Group chat iteration completed or interrupted: {str(e)}")
            print("🔄 Proceeding with message processing...")

        # Retrieve the final chat history - handle async generator properly
        print("📊 Retrieving messages from chat history...")
        messages = []
        try:
            # Convert async generator to list
            chat_messages = group_chat.get_chat_messages()
            if hasattr(chat_messages, '__aiter__'):
                # It's an async generator
                async for msg in chat_messages:
                    messages.append(msg)
            else:
                # It's already a list or iterable
                messages = list(chat_messages)
        except Exception as e:
            print(f"⚠️ Error retrieving chat messages: {str(e)}")
            print("🔄 Continuing with empty message list...")
            messages = []
        
        print(f"📊 Retrieved {len(messages)} messages from chat history.")

        # 1️⃣ Check if the Product Owner says "READY FOR USER APPROVAL"
        approval_requested = False
        for msg in messages:
            if (
                isinstance(msg, ChatMessageContent) and
                msg.role == AuthorRole.ASSISTANT and
                "READY FOR USER APPROVAL" in msg.content.upper()
            ):
                approval_requested = True
                print("📝 The Product Owner says 'READY FOR USER APPROVAL'.")
                print("💡 Type 'APPROVED' to finalize, 'SKIP' to terminate without approval, or anything else to cancel.")
                
                try:
                    user_input = input("Your response: ").strip().upper()
                    
                    if user_input == "APPROVED":
                        final_user_message = ChatMessageContent(
                            role=AuthorRole.USER,
                            content="APPROVED"
                        )
                        await group_chat.add_chat_message(final_user_message)
                        print("✅ Final user approval added.")
                    elif user_input == "SKIP":
                        print("⚠️ User chose to skip approval. Proceeding with HTML extraction only.")
                        break
                    else:
                        print("⚠️ Approval not given. Proceeding with HTML extraction only.")
                        break
                        
                except (KeyboardInterrupt, EOFError):
                    print("\n⚠️ User interrupted. Proceeding with HTML extraction only.")
                    break
                break

        # 2️⃣ Check if the final "APPROVED" is in the chat history
        # Re-fetch messages after potential approval
        if approval_requested:
            print("🔄 Re-fetching messages after potential approval...")
            try:
                chat_messages = group_chat.get_chat_messages()
                if hasattr(chat_messages, '__aiter__'):
                    messages = []
                    async for msg in chat_messages:
                        messages.append(msg)
                else:
                    messages = list(chat_messages)
            except Exception as e:
                print(f"⚠️ Error re-fetching messages: {str(e)}")

        user_approved = False
        for msg in messages:
            if (
                isinstance(msg, ChatMessageContent) and
                msg.role == AuthorRole.USER and
                "APPROVED" in msg.content.upper()
            ):
                user_approved = True
                print("✅ User approval confirmed in chat history.")
                await on_approved_callback()
                break

        if not user_approved and approval_requested:
            print("ℹ️ No final approval given. Skipping GitHub push.")

        # 3️⃣ Extract HTML code from Software Engineer's messages
        html_code = None
        print(f"🔍 Searching through {len(messages)} messages for HTML code...")

        for i, msg in enumerate(messages):
            if isinstance(msg, ChatMessageContent) and msg.role == AuthorRole.ASSISTANT:
                author_name = getattr(msg, "author_name", "Unknown")
                print(f"📄 Message {i+1}: Author={author_name}, Content preview: {msg.content[:100]}...")

                # Check if this is from SoftwareEngineer or contains HTML
                if (author_name == "SoftwareEngineer" or "html" in msg.content.lower()):
                    print(f"🎯 Found potential HTML message from {author_name}")

                    # Try multiple HTML extraction patterns
                    patterns = [
                        r"```html\s*(.*?)```",           # Standard ```html block
                        r"```HTML\s*(.*?)```",           # Uppercase HTML
                        r"```\s*html\s*(.*?)```",        # html with spaces
                        r"```\s*(<!DOCTYPE html.*?)```", # HTML starting with DOCTYPE
                        r"```\s*(<html.*?</html>)```",   # HTML tags
                        r"```\s*(.*?)</html>\s*```"      # Content ending with </html>
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, msg.content, re.DOTALL | re.IGNORECASE)
                        if match:
                            html_code = match.group(1).strip()
                            print(f"✅ Extracted HTML using pattern: {pattern}")
                            print(f"📄 HTML preview: {html_code[:200]}...")
                            break

                    if html_code:
                        break

        if html_code:
            try:
                output_path = os.path.join(os.getcwd(), "index.html")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_code)
                print(f"✅ HTML code saved to: {output_path}")
                print(f"📁 File size: {len(html_code)} characters")

                # Try to open in browser
                try:
                    webbrowser.open(f"file://{output_path}")
                    print("🌐 Opened in default browser!")
                except Exception as browser_error:
                    print(f"⚠️ Could not open browser: {browser_error}")
            except Exception as e:
                print(f"❌ Error writing to index.html: {e}")
        else:
            print("⚠️ No HTML code block found from any agent.")
            print("💡 Try asking the Software Engineer to provide HTML code in a ```html code block.")

            # 🔧 Fallback: Look for any HTML-like content (even without code blocks)
            if messages:  # Only try fallback if we have messages
                print("🔍 Searching for HTML-like content as fallback...")
                for i, msg in enumerate(messages):
                    if isinstance(msg, ChatMessageContent) and msg.role == AuthorRole.ASSISTANT:
                        content = msg.content.lower()
                        if any(tag in content for tag in ["<html", "<!doctype", "<head", "<body"]):
                            print(f"📄 Found HTML-like content in message {i+1}")
                            # Extract potential HTML content
                            html_match = re.search(r'(<!DOCTYPE.*?</html>|<html.*?</html>)', msg.content, re.DOTALL | re.IGNORECASE)
                            if html_match:
                                fallback_html = html_match.group(1).strip()
                                try:
                                    output_path = os.path.join(os.getcwd(), "index.html")
                                    with open(output_path, "w", encoding="utf-8") as f:
                                        f.write(fallback_html)
                                    print(f"✅ Fallback HTML saved to: {output_path}")
                                    print(f"📁 File size: {len(fallback_html)} characters")

                                    try:
                                        webbrowser.open(f"file://{output_path}")
                                        print("🌐 Opened in default browser!")
                                    except Exception as browser_error:
                                        print(f"⚠️ Could not open browser: {browser_error}")
                                    break
                                except Exception as e:
                                    print(f"❌ Error writing fallback HTML: {e}")

        print("🏁 Multi-agent workflow completed successfully!")
        return messages

    except Exception as e:
        print(f"❌ Error in run_multi_agent: {str(e)}")
        return []
    finally:
        # Ensure proper cleanup
        await asyncio.sleep(0.1)  # Small delay to allow cleanup

# --- Async main function with proper cleanup
async def main():
    """Main async function with proper event loop handling."""
    try:
        print("🚀 Multi-Agent System Starting...")
        user_input = input("📝 Please enter your prompt for the multi-agent system: ")
        if user_input.strip():
            await run_multi_agent(user_input)
        else:
            print("❌ Empty input provided. Exiting.")
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted the process. Exiting gracefully.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("🔍 This might be due to API limits, network issues, or configuration problems.")
    finally:
        # Give time for cleanup
        await asyncio.sleep(0.1)

# --- For running directly with proper event loop handling
if __name__ == "__main__":
    try:
        # Use asyncio.run() which handles event loop creation and cleanup properly
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            print("⚠️ Event loop cleanup completed.")
        else:
            print(f"❌ Runtime error: {str(e)}")
    except Exception as e:
        print(f"❌ Final error: {str(e)}")
    finally:
        # Ensure all pending tasks are cancelled
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Give tasks time to cancel
                    loop.run_until_complete(asyncio.sleep(0.1))
        except Exception:
            # Ignore cleanup errors
            pass
        print("🔚 Application terminated.")