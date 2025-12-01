import json
import re
import os
from sentence_transformers import SentenceTransformer, util
from google import genai
from environs import Env
from langchain.memory import ConversationBufferMemory

# Initialize environment, Gemini client, and conversation memory.
env = Env()
env.read_env()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Replace with actual key
client = genai.Client(api_key=GEMINI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global counters and session variables.
conversation_turn = 0
reference_context = None
global_baseline_emotion = "Unknown"
global_target_emotion = "Relief"

# Emotion transition mapping.
target_emotion_map = {
    "Guilt": "Self-Forgiveness",
    "Exhaustion": "Restoration",
    "Overthinking": "Clarity",
    "Apathy": "Motivation",
    "Shame": "Self-Acceptance",
    "Isolation": "Reconnection",
    "Self-Doubt": "Confidence",
    "Regret": "Hope",
    "Stress": "Resilience",
    "Resentment": "Understanding",
    "Anxiety": "Calmness",
    "Heartbreak": "Healing",
    "Frustration": "Patience",
    "Loneliness": "Connection",
    "Nervousness": "Preparedness",
    "Pressure": "Confidence",
    "Hopelessness": "Purpose",
    "Overwhelm": "Clarity",
    "Worry": "Reassurance",
    "Hesitation": "Clarity",
    "Uncertainty": "Trust",
    "Sadness": "Hope",
    "Insecurity": "Self-Worth",
    "Disappointment": "Resilience",
    "Resistance": "Openness",
    "Confusion": "Understanding",
    "Comparison": "Self-Appreciation",
    "Discouragement": "Determination",
    "Doubt": "Conviction",
    "Burnout": "Clarity",
    "Hurt": "Forgiveness"
}

def parse_conversation(conversation_str):
    header_pattern = r"^(?:Here is the conversation:|Here is a structured student counseling session's conversation:|Here is the counseling session conversation:)\s*"
    conversation_str = re.sub(header_pattern, "", conversation_str, flags=re.IGNORECASE)
    pattern = r"\*\*(Student|Counselor|Final Advice)\*\*:\s*(.*?)(?=\n\*\*(Student|Counselor|Final Advice)\*\*:|$)"
    matches = re.findall(pattern, conversation_str, re.DOTALL)
    turns = []
    for speaker, text, _ in matches:
        cleaned_text = text.strip().replace("\n", " ")
        turns.append({"speaker": speaker, "text": cleaned_text})
    return turns

def extract_student_statement(session):
    if "student_first_statement" in session and session["student_first_statement"].strip():
        return session["student_first_statement"].strip()
    conversation_field = session.get("conversation", "")
    if isinstance(conversation_field, str):
        turns = parse_conversation(conversation_field)
        for turn in turns:
            if turn.get("speaker", "").lower() == "student":
                return turn.get("text", "").strip()
    return ""

def load_dataset_entries(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    entries = []
    for session in data:
        student_text = extract_student_statement(session)
        if not student_text:
            continue
        embedding = st_model.encode(student_text, convert_to_tensor=True)
        entry = {
            "session_id": session.get("session_id"),
            "student_emotion_before": session.get("student_emotion_before", ""),
            "student_emotion_after": session.get("student_emotion_after", ""),
            "primary_issue": session.get("primary_issue", ""),
            "conversation": session.get("conversation", ""),
            "text": student_text,
            "embedding": embedding
        }
        entries.append(entry)
    return entries

def build_context_from_entries(entries):
    context_list = []
    for entry in entries:
        snippet = entry["text"][:300]
        context_list.append(
            f"Session ID: {entry['session_id']}\n"
            f"Issue: {entry['primary_issue']}\n"
            f"Emotion Before: {entry['student_emotion_before']}\n"
            f"Emotion After: {entry['student_emotion_after']}\n"
            f"Excerpt: {snippet}..."
        )
    return "\n\n".join(context_list)

def retrieve_relevant_entries(student_query, dataset_entries, top_k=5):
    query_embedding = st_model.encode(student_query, convert_to_tensor=True)
    similarities = [(util.cos_sim(query_embedding, entry["embedding"]).item(), entry) for entry in dataset_entries]
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    top_entries = [entry for _, entry in similarities[:top_k]]
    return top_entries

def update_memory(student_input, bot_response):
    memory.chat_memory.add_user_message(student_input)
    memory.chat_memory.add_ai_message(bot_response)

def get_formatted_conversation():
    return memory.load_memory_variables({})["chat_history"]

def build_gemini_prompt(student_query, context_str, past_conversation, baseline_emotion, target_emotion, turn_number):
    # Conversation phase-based instructions:
    if turn_number <= 2:
        phase_instruction = ("In these initial turns, focus on understanding the student's issue deeply. "
                             "Ask reflective, open-ended questions to explore their feelings and situation, without offering solutions yet.")
    elif turn_number <= 4:
        phase_instruction = ("Now begin to gently introduce practical suggestions and guidance, "
                             "while still acknowledging the student's struggles and emotions.")
    else:
        phase_instruction = ("Provide a final, summarizing optimistic advice that encourages hope and actionable next steps, "
                             "emphasizing the student's strengths and progress.")
    
    prompt_text = (
        "You are an empathetic and insightful virtual psychiatrist. Your goal is to understand the student's emotional state "
        "and gently guide them toward their target emotional state.\n\n"
        f"---\nBaseline Emotion: {baseline_emotion}\nTarget Emotion: {target_emotion}\n---\n\n"
        "🧠 Relevant Past Cases (reference from dataset):\n"
        f"{context_str}\n\n"
        "🗃️ Ongoing Conversation:\n"
        f"{past_conversation}\n\n"
        "💬 Conversation Guidance:\n"
        f"{phase_instruction}\n\n"
        "Respond in 3-4 concise sentences using a tone similar to the following examples:\n"
        "• Initially, deeply explore the student's issues and emotions with reflective questions.\n"
        "• Later, introduce gentle suggestions and guidance.\n"
        "• Finally, provide a summarizing optimistic advice.\n\n"
        f"Student says: {student_query}\n\n"
        "How should you respond?"
    )
    return [
        {"role": "model", "parts": [{"text": prompt_text}]},
        {"role": "user", "parts": [{"text": student_query}]}
    ]

reference_context = None

def generate_response_with_rag(student_query, dataset_entries, turn_number):
    global reference_context, global_baseline_emotion, global_target_emotion

    # retrieve reference entries based on the 1st dialogue
    if turn_number == 1:
        top_entries = retrieve_relevant_entries(student_query, dataset_entries, top_k=5)
        reference_context = build_context_from_entries(top_entries)
        if top_entries:
            global_baseline_emotion = top_entries[0].get("student_emotion_before", "Unknown")
            global_target_emotion = target_emotion_map.get(global_baseline_emotion, "Relief")
        else:
            global_baseline_emotion = "Unknown"
            global_target_emotion = "Relief"
    
    past_conversation = get_formatted_conversation()
    final_prompt = build_gemini_prompt(
        student_query=student_query,
        context_str=reference_context,
        past_conversation=past_conversation,
        baseline_emotion=global_baseline_emotion,
        target_emotion=global_target_emotion,
        turn_number=turn_number
    )

    response = client.models.generate_content_stream(
        model="gemini-1.5-pro",
        contents=final_prompt,
    )
    
    response_text = ""
    for chunk in response:
        response_text += chunk.text
    update_memory(student_query, response_text)
    return response_text.strip()

def chatbot_interface():
    global conversation_turn
    DATASET_PATH = "student_psychiatric_sessions_balanced (3).json"
    dataset_entries = load_dataset_entries(DATASET_PATH)
    print(f"Loaded {len(dataset_entries)} session entries from dataset.\n")
    
    print("Welcome to the Psychiatric Chatbot Interface!")
    initial_issue = input("Please describe your issue: ")
    conversation_turn += 1
    initial_response = generate_response_with_rag(initial_issue, dataset_entries, conversation_turn)
    print("\nChatbot:", initial_response, "\n")
    
    while True:
        student_query = input("Student: ")
        if student_query.lower().strip() == "exit":
            print("Exiting chat. Take care!")
            break
        conversation_turn += 1
        response = generate_response_with_rag(student_query, dataset_entries, conversation_turn)
        print("\nChatbot:", response, "\n")

if __name__ == "__main__":
    chatbot_interface()
