#!/usr/bin/env python3
"""
🚀 RUN CHATBOT INTERFACE
========================

Quick launcher for the AI Ecosystem Assistant chatbot.
Provides voice-capable AI with access to all 386+ revolutionary tools.

Usage:
    python run_chatbot.py --web    # Start web interface
    python run_chatbot.py --cli    # Start command line interface

Features:
✅ Voice Input & Output
✅ Natural Language Processing
✅ 386+ Integrated Tools
✅ Real-time Responses
✅ Consciousness-Enhanced AI
✅ Quantum-Safe Security
"""

import sys
import os

def main():
    """Main launcher function"""
    print("🚀 AI Ecosystem Assistant Launcher")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_chatbot.py --web    # Web interface with voice")
        print("  python run_chatbot.py --cli    # Command line interface")
        print()
        print("🌐 Web Interface Features:")
        print("   ✅ Voice input/output")
        print("   ✅ Modern UI/UX")
        print("   ✅ Real-time responses")
        print("   ✅ Tool integration")
        print()
        print("💻 CLI Interface Features:")
        print("   ✅ Text-based interaction")
        print("   ✅ Fast responses")
        print("   ✅ All tools available")
        return

    mode = sys.argv[1]

    if mode == '--web':
        print("🌐 Starting Web Interface...")
        print("🎯 Features: Voice • Modern UI • Real-time • 386+ Tools")
        print()
        print("📱 Visit: http://localhost:5000")
        print("=" * 50)

        # Import and run web interface
        try:
            from PRODUCT_CHATBOT_INTERFACE import ProductChatbotInterface
            chatbot = ProductChatbotInterface()
            chatbot.run_web_server()
        except ImportError as e:
            print(f"❌ Import Error: {e}")
            print("💡 Install required packages:")
            print("   pip install flask flask-cors flask-socketio")
            print("   pip install speechrecognition pyttsx3")
            print("   pip install torch transformers")

    elif mode == '--cli':
        print("💻 Starting CLI Interface...")
        print("🎯 Features: Fast • Text-based • All Tools")
        print("=" * 50)

        # Simple CLI interface
        try:
            from PRODUCT_CHATBOT_INTERFACE import ProductChatbotInterface
            chatbot = ProductChatbotInterface()

            print("🤖 AI Ecosystem Assistant (CLI Mode)")
            print("Type 'quit' to exit")
            print("-" * 40)

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 Goodbye! Your AI Ecosystem Assistant is always here to help.")
                    break

                if user_input:
                    # Process the message
                    response = chatbot.process_message(user_input, "cli_session")

                    print(f"🤖 Assistant: {response['text']}")

                    if response.get('tools_used'):
                        print(f"🔧 Tools used: {', '.join(response['tools_used'])}")

                    print("-" * 40)

        except KeyboardInterrupt:
            print("\n🤖 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")

    else:
        print(f"❌ Unknown mode: {mode}")
        print("Use --web or --cli")

if __name__ == "__main__":
    main()
