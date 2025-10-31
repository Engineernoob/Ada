#!/usr/bin/env python3
"""
Ada Cloud Client - Interact with Ada running on Modal Cloud
"""

import asyncio
import json
import subprocess
import sys
from typing import Dict, Any, Optional

class AdaClient:
    """Client for interacting with Ada on Modal Cloud."""
    
    def __init__(self):
        """Initialize the Ada Cloud client."""
        print("🌩️  Connecting to Ada Cloud...")
        self.app_name = "AdaCloudFinal"
        print("✅ Cloud client initialized!")
    
    async def _run_modal_function(self, function_name: str, data: str = None) -> Dict[str, Any]: # pyright: ignore[reportArgumentType]
        """Run a Modal function via CLI."""
        try:
            cmd = ["modal", "run", f"cloud.modal_app::{function_name}"]
            if data:
                cmd.extend(["--data", data])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse the output for JSON if possible
                try:
                    # Try to extract JSON from output
                    output = result.stdout.strip()
                    if output:
                        return {"success": True, "response": output}
                except:
                    pass
                return {"success": True, "response": result.stdout}
            else:
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def chat(self, message: str) -> str:
        """Send a message to Ada and get response."""
        print(f"🤖 You: {message}")
        
        try:
            # Prepare data for inference
            data = {
                "prompt": message,
                "module": "core.reasoning",
                "parameters": {
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            }
            
            # Call the cloud inference function
            result = await self._run_modal_function("ada_infer", json.dumps(data))
            
            if result.get("success"):
                response = result.get("response", "I apologize, but I couldn't generate a response.")
                print(f"🤖 Ada: {response}")
                return response
            else:
                error = result.get("error", "Unknown error occurred")
                print(f"❌ Ada Error: {error}")
                return f"Error: {error}"
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return f"Connection error: {e}"
    
    async def run_mission(self, goal: str) -> Dict[str, Any]:
        """Run a mission on Ada Cloud."""
        print(f"🎯 Starting Mission: {goal}")
        
        try:
            result = await self.mission_func.aio.call(goal) # type: ignore
            
            if result.get("success"):
                print(f"✅ Mission completed successfully!")
                print(f"   Status: {result.get('mission_status')}")
                print(f"   Steps completed: {result.get('completed_steps', 0)}")
                print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
                return result
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ Mission failed: {error}")
                return {"success": False, "error": error}
                
        except Exception as e:
            print(f"❌ Mission error: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize(self, target_module: str = "core.reasoning") -> Dict[str, Any]:
        """Run optimization on Ada Cloud."""
        print(f"🔧 Optimizing module: {target_module}")
        
        try:
            params = {
                "target_module": target_module,
                "parameter_space": {
                    "learning_rate": {"type": "float", "min": 0.001, "max": 0.1},
                    "batch_size": {"type": "int", "min": 16, "max": 128}
                },
                "max_iterations": 5,
                "algorithm": "random_search"
            }
            
            result = await self.optimize_func.aio.call(json.dumps(params)) # pyright: ignore[reportAttributeAccessIssue]
            
            if result.get("success"):
                print(f"✅ Optimization completed!")
                print(f"   Best score: {result.get('best_score', 0)}")
                print(f"   Improvement: {result.get('improvement', 0):.2%}")
                print(f"   Iterations: {result.get('iterations', 0)}")
                return result
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ Optimization failed: {error}")
                return {"success": False, "error": error}
                
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return {"success": False, "error": str(e)}
    
    async def save_conversation(self, conversation: list) -> bool:
        """Save conversation to cloud storage."""
        try:
            filename = f"conversations/chat_{asyncio.get_event_loop().time()}.json"
            result = await self.upload_json_func.aio.call(filename, conversation) # type: ignore
            return result.get("success", False)
        except Exception as e:
            print(f"❌ Failed to save conversation: {e}")
            return False
    
    async def interactive_mode(self):
        """Run interactive chat mode."""
        print("\n🚀 Ada Cloud Interactive Mode")
        print("Commands: /mission <goal>, /optimize, /help, /quit")
        print("=" * 50)
        
        conversation = []
        
        while True:
            try:
                user_input = input("\n🤖 You: ").strip()
                
                if user_input.lower() in ['quit', '/quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/help':
                    print("\n📖 Ada Cloud Commands:")
                    print("  /mission <goal>  - Run Ada mission")
                    print("  /optimize        - Optimize Ada performance")
                    print("  /help           - Show this help")
                    print("  /quit           - Exit")
                    continue
                
                elif user_input.startswith('/mission '):
                    goal = user_input[9:].strip()
                    if goal:
                        await self.run_mission(goal)
                    else:
                        print("❌ Please provide a mission goal")
                    continue
                
                elif user_input.lower() == '/optimize':
                    await self.optimize()
                    continue
                
                # Regular chat message
                if user_input:
                    conversation.append({"type": "user", "message": user_input})
                    
                    response = await self.chat(user_input)
                    conversation.append({"type": "ada", "message": response})
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
        
        # Save conversation
        if conversation:
            await self.save_conversation(conversation)
            print(f"💾 Conversation saved to cloud storage")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "chat":
            client = AdaClient()
            if len(sys.argv) > 2:
                message = " ".join(sys.argv[2:])
                await client.chat(message)
            else:
                await client.interactive_mode()
        
        elif command == "mission":
            if len(sys.argv) > 2:
                goal = " ".join(sys.argv[2:])
                client = AdaClient()
                await client.run_mission(goal)
            else:
                print("❌ Please provide a mission goal")
                print("Usage: python cloud_client.py mission \"<goal>\"")
        
        elif command == "optimize":
            client = AdaClient()
            await client.optimize()
        
        else:
            print("❌ Unknown command")
            print("Available commands: chat, mission, optimize")
    else:
        client = AdaClient()
        await client.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
