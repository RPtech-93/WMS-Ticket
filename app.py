"""WMS WhatsApp Ticket Logging - Flask webhook + Supabase + Gemini AI."""

from flask import Flask, request, jsonify
import os, re, json, requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
app = Flask(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

MESSAGE_FIELDS = MANDATORY_FIELDS = [
    "Order ID", "Warehouse", "Module", "Screen", "Category",
    "Problem Description", "Error Message", "Business Impact", "Priority",
]
SLA_HOURS = {"high": 4, "medium": 12, "low": 24}
DEFAULT_ASSIGNED_TEAM = "WMS Functional Team (Benchmark)"

# In-memory state (resets on server restart)
processed_message_ids = set()          # Meta webhook retries -> avoid duplicate tickets/replies
pending_tickets = {}                   # sender -> {"fields", "updated_at"} for missing-field follow-ups
last_help_sent = {}                    # sender -> last time we sent the format-guide message
PENDING_TICKET_TIMEOUT_MINUTES = 30
HELP_MESSAGE_COOLDOWN_MINUTES = 15


# ---------------------------------------------------------------- helpers

def get_next_ticket_id():
    count = supabase.table("tickets").select("ticket_id", count="exact").execute().count or 0
    return f"WMS-{datetime.now().year}-{count + 1:06d}"


def parse_ticket_message(message_text):
    """Extract fields; each value stops at the next known field label (handles missing newlines)."""
    fields = {k: "" for k in MESSAGE_FIELDS}
    alt = "|".join(re.escape(k) for k in MESSAGE_FIELDS)
    for key in fields:
        m = re.search(rf"{re.escape(key)}\s*:\s*(.*?)(?=\s*(?:{alt})\s*:|$)", message_text, re.IGNORECASE | re.DOTALL)
        if m:
            fields[key] = m.group(1).strip()
    return fields


def get_missing_fields(fields):
    return [k for k in MANDATORY_FIELDS if not fields.get(k, "").strip()]


def calculate_sla_due_date(priority, from_time):
    return (from_time + timedelta(hours=SLA_HOURS.get(priority.strip().lower(), 24))).isoformat()


def log_ticket(warehouse_source, sender_name, sender_number, fields, duplicate_ticket_id=None):
    now = datetime.now()
    ticket_id = get_next_ticket_id()
    priority = fields["Priority"] or "Medium"
    row = {
        "ticket_id": ticket_id,
        "order_id": fields["Order ID"],
        "warehouse": fields["Warehouse"] or warehouse_source,
        "raised_by": sender_name,
        "contact_number": sender_number,
        "date": now.strftime("%d-%b-%Y"),
        "time": now.strftime("%H:%M"),
        "module": fields["Module"],
        "screen": fields["Screen"],
        "category": fields["Category"],
        "problem_description": fields["Problem Description"],
        "error_message": fields["Error Message"],
        "business_impact": fields["Business Impact"],
        "priority": priority,
        "ai_intent": "Duplicate/Repeat Issue" if duplicate_ticket_id else "New Issue",
        "duplicate_of": duplicate_ticket_id or "",
        "affected_user_count": "1",
        "assigned_team": DEFAULT_ASSIGNED_TEAM,
        "assigned_to": "",
        "status": "New",
        "sla_due_date": calculate_sla_due_date(priority, now),
        "resolved_date": None,
        "closed_date": None,
        "resolution": "",
        "user_confirmation": "Pending",
        "remarks": "",
    }
    supabase.table("tickets").insert(row).execute()
    return ticket_id


# ------------------------------------------------------------- Gemini AI

def call_gemini_json(prompt, label="Gemini", max_attempts=2):
    """Calls Gemini, retries transient failures, prints full error body on failure."""
    if not GEMINI_API_KEY:
        print(f"[{label}] WARNING: GEMINI_API_KEY missing")
        return None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                GEMINI_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=10,
                proxies={"http": None, "https": None},  # bypass any auto-detected system proxy
            )
        except Exception as e:
            print(f"[{label}] Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                continue
            return None

        if resp.status_code != 200:
            print(f"[{label}] Attempt {attempt}/{max_attempts} non-200 ({resp.status_code}): {resp.text}")
            if attempt < max_attempts:
                continue
            return None

        result = resp.json()
        if "candidates" not in result:
            print(f"[{label}] No 'candidates' in response: {result}")
            return None
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except Exception as e:
            print(f"[{label}] Parse failure: {e}, raw: {result}")
            return None
    return None


def ai_fill_missing_fields(message_text, fields):
    """Fills only the MISSING fields via AI. On any failure, returns fields unchanged (never silently drops)."""
    missing = [k for k in MESSAGE_FIELDS if not fields.get(k, "").strip()]
    if not missing:
        return fields

    prompt = (
        "Neeche ek WhatsApp message hai jisme warehouse/WMS problem describe hai. "
        f"Isme se ye fields nikaalo agar mention hain: {', '.join(missing)}.\n\n"
        "Context se reasonably infer karo (e.g. 'GRN nahi ho pa raha' -> Module='Inbound', "
        "Category='GRN'). Priority clear na ho to urgency se estimate karo. Genuinely na pata "
        "chale to empty string rakho - hallucinate mat karo.\n\n"
        f"Message:\n{message_text}\n\n"
        "Sirf JSON return karo: {" + ", ".join(f'"{f}": ""' for f in missing) + "}"
    )
    ai_fields = call_gemini_json(prompt, label="AI Fill Missing")
    if not ai_fields:
        return fields
    merged = dict(fields)
    for key, value in ai_fields.items():
        if key in MESSAGE_FIELDS and value and not merged.get(key, "").strip():
            merged[key] = value.strip()
    return merged


def ai_classify_and_extract(message_text):
    """One AI call: decides if message is genuinely a ticket (any wording/spelling), and extracts fields."""
    empty = {"is_ticket": False, **{k: "" for k in MESSAGE_FIELDS}}
    if not GEMINI_API_KEY:
        return empty

    prompt = (
        "WMS WhatsApp message classifier. Decide:\n"
        "1. is_ticket: genuinely ek warehouse/WMS problem/request hai kya? (koi bhi keyword/spelling "
        "chale - '#query', 'query', 'qeury', ya bina keyword ke bhi). Greetings/casual chat = false.\n"
        f"2. Agar true, extract: {', '.join(MESSAGE_FIELDS)} (infer context se, empty rakho agar na pata chale, hallucinate mat karo).\n\n"
        f"Message:\n{message_text}\n\n"
        "JSON only: {\"is_ticket\": true/false, " + ", ".join(f'"{f}": ""' for f in MESSAGE_FIELDS) + "}"
    )
    parsed = call_gemini_json(prompt, label="AI Classify")
    if not parsed:
        return empty
    out = dict(empty)
    out["is_ticket"] = bool(parsed.get("is_ticket", False))
    for key in MESSAGE_FIELDS:
        if parsed.get(key):
            out[key] = str(parsed[key]).strip()
    return out


def ai_check_duplicate(fields):
    """Compares against recent open tickets in the same warehouse. Returns matching ticket_id or None."""
    if not GEMINI_API_KEY:
        return None
    try:
        recent = (supabase.table("tickets").select("ticket_id, problem_description, error_message")
                  .eq("warehouse", fields.get("Warehouse", "")).eq("status", "New")
                  .order("created_at", desc=True).limit(10).execute())
        if not recent.data:
            return None
        existing = "\n".join(f"- {t['ticket_id']}: {t['problem_description']} ({t['error_message']})" for t in recent.data)
        prompt = (
            "Naya WMS ticket aur usi warehouse ke recent open tickets neeche hain. Batao kya naya "
            "ticket in mein se kisi ka duplicate/repeat hai (same underlying problem).\n\n"
            f"Naya: {fields.get('Problem Description', '')} ({fields.get('Error Message', '')})\n\n"
            f"Existing:\n{existing}\n\n"
            'JSON only: {"is_duplicate": true/false, "duplicate_ticket_id": ""}'
        )
        parsed = call_gemini_json(prompt, label="AI Duplicate Check")
        if parsed and parsed.get("is_duplicate") and parsed.get("duplicate_ticket_id"):
            return parsed["duplicate_ticket_id"]
        return None
    except Exception as e:
        print(f"[AI Duplicate Check Error] {e}")
        return None


# ------------------------------------------------------------- WhatsApp

def send_whatsapp_message(to_number, message_text):
    if not WHATSAPP_ACCESS_TOKEN:
        print("WARNING: WHATSAPP_ACCESS_TOKEN missing")
        return None
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to_number, "type": "text", "text": {"body": message_text}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"[WhatsApp Send] Status: {resp.status_code}, Response: {resp.text}")
        return resp.json()
    except Exception as e:
        print(f"[WhatsApp Send Error] {e}")
        return None


def show_typing_indicator(message_id):
    """
    Marks the incoming message as read (blue ticks) and shows a "typing..."
    indicator to the sender while we process (AI calls, DB writes, etc).
    WhatsApp auto-dismisses it after we reply, or after ~25 seconds -
    whichever comes first. Best-effort only: if this fails, we just skip it
    and continue processing normally (never blocks the actual ticket logic).
    """
    if not WHATSAPP_ACCESS_TOKEN or not message_id:
        return
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }
    try:
        requests.post(url, headers=headers, json=payload, timeout=5)
    except Exception as e:
        print(f"[Typing Indicator Error] {e}")


HELP_TEXT = (
    "Hi! This number is for logging WMS support tickets.\n\n"
    "To raise a ticket, please send a message with these details:\n\n"
    "Order ID: <order number>\nWarehouse: <warehouse name>\nModule: <e.g. Inbound / Outbound>\n"
    "Screen: <screen name>\nCategory: <issue category>\nProblem Description: <what's happening>\n"
    "Error Message: <exact error, if any>\nBusiness Impact: <how it's affecting work>\n"
    "Priority: <High / Medium / Low>"
)


# --------------------------------------------------------------- routes

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge"), 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        raw_data = request.get_json(force=True, silent=True) or {}
        print(f"[RouteMobile Raw Payload]: {json.dumps(raw_data)}")
    except Exception as e:
        print(f"[Log Error]: {e}")
        return jsonify({"status": "error", "reason": "invalid json"}), 200
 
    try:
        # 1. Agar ye Delivery Status (MT) hai, toh bas 200 OK bhej do
        if "statuses" in raw_data and "entry" not in raw_data:
            return jsonify({"status": "ignored", "reason": "status update event"}), 200
 
        message = None
        contacts = []
        phone_number_id = PHONE_NUMBER_ID
 
        if "entry" in raw_data:
            value = raw_data["entry"][0]["changes"][0]["value"]
            if "statuses" in value:
                return jsonify({"status": "ignored", "reason": "status update event inside entry"}), 200
            messages = value.get("messages", [])
            if not messages:
                return jsonify({"status": "ignored", "reason": "no message content"}), 200
            message = messages[0]
            contacts = value.get("contacts", [])
            phone_number_id = value.get("metadata", {}).get("phone_number_id", PHONE_NUMBER_ID)
        elif "messages" in raw_data:
            messages = raw_data.get("messages", [])
            if not messages:
                return jsonify({"status": "ignored", "reason": "no message content"}), 200
            message = messages[0]
            contacts = raw_data.get("contacts", [])
            phone_number_id = raw_data.get("brand_msisdn", PHONE_NUMBER_ID)
        else:
            return jsonify({"status": "ignored", "reason": "unknown payload structure"}), 200
 
        msg_id = message.get("message_id") or message.get("id")
        if msg_id and msg_id in processed_message_ids:
            print(f"[Duplicate Webhook] {msg_id} already processed, skipping")
            return jsonify({"status": "ignored", "reason": "duplicate webhook delivery"}), 200
        if msg_id:
            processed_message_ids.add(msg_id)
            show_typing_indicator(msg_id)
 
        message_text = message.get("text", {}).get("body", "")
        sender_number = message.get("from", "Unknown Number")
        sender_name = contacts[0].get("profile", {}).get("name", sender_number) if contacts else sender_number
 
        if not message_text:
            return jsonify({"status": "ignored", "reason": "empty message"}), 200
 
        # --- TICKET LOGGING & AI LOGIC ---
        fields = parse_ticket_message(message_text)
        looks_like_ticket = message_text.strip().startswith("#") or any(v.strip() for v in fields.values())
 
        now = datetime.now()
        pending = pending_tickets.get(sender_number)
        if pending and looks_like_ticket and (now - pending["updated_at"]).total_seconds() <= PENDING_TICKET_TIMEOUT_MINUTES * 60:
            conflict = any(
                pending["fields"].get(f, "").strip() and fields.get(f, "").strip()
                and pending["fields"][f].strip().lower() != fields[f].strip().lower()
                for f in ("Order ID", "Warehouse")
            )
            if conflict:
                pending_tickets.pop(sender_number, None)
            else:
                merged = dict(pending["fields"])
                merged.update({k: v for k, v in fields.items() if v.strip()})
                fields = merged
                looks_like_ticket = True
 
        missing = get_missing_fields(fields)
 
        if missing and looks_like_ticket:
            fields = ai_fill_missing_fields(message_text, fields)
            missing = get_missing_fields(fields)
        elif missing and not looks_like_ticket:
            ai_result = ai_classify_and_extract(message_text)
            if not ai_result["is_ticket"]:
                last_sent = last_help_sent.get(sender_number)
                if not last_sent or (now - last_sent).total_seconds() > HELP_MESSAGE_COOLDOWN_MINUTES * 60:
                    send_whatsapp_message(sender_number, HELP_TEXT)
                    last_help_sent[sender_number] = now
                return jsonify({"status": "ignored", "reason": "not a ticket (AI classified)"}), 200
            for key in MESSAGE_FIELDS:
                if not fields.get(key, "").strip() and ai_result.get(key):
                    fields[key] = ai_result[key]
            missing = get_missing_fields(fields)
 
        if missing:
            pending_tickets[sender_number] = {"fields": fields, "updated_at": now}
            reply = (
                "⚠️ Your ticket could not be created. The following required fields are missing:\n\n"
                + "\n".join(f"- {f}" for f in missing)
                + "\n\nYou can reply with just the missing information, or resend your message with the complete format."
            )
            send_whatsapp_message(sender_number, reply)
            return jsonify({"status": "rejected", "reason": "missing mandatory fields", "missing_fields": missing}), 200
 
        pending_tickets.pop(sender_number, None)
        duplicate_id = ai_check_duplicate(fields)
        ticket_id = log_ticket(phone_number_id, sender_name, sender_number, fields, duplicate_id)
 
        confirmation = f"✅ Your ticket has been successfully created.\nTicket ID: {ticket_id}"
        if duplicate_id:
            confirmation = (
                f"✅ Your ticket has been created.\nTicket ID: {ticket_id}\n\n"
                f"⚠️ Note: This issue looks similar to an already open ticket ({duplicate_id}) — "
                "our team will review both together."
            )
        send_whatsapp_message(sender_number, confirmation)
 
        return jsonify({"status": "logged", "ticket_id": ticket_id, "duplicate_of": duplicate_id,
                         "message": "Ticket successfully logged"}), 200
 
    except Exception as err:
        print(f"[Webhook Execution Error]: {err}")
        return jsonify({"status": "error", "reason": str(err)}), 200


@app.route("/api/tickets", methods=["GET"])
def get_tickets():
    """All tickets, optionally filtered: ?warehouse=&status=&priority="""
    query = supabase.table("tickets").select("*")
    for param, column in (("warehouse", "warehouse"), ("status", "status"), ("priority", "priority")):
        value = request.args.get(param)
        if value:
            query = query.eq(column, value)
    tickets = query.order("created_at", desc=True).execute().data
    return jsonify({"count": len(tickets), "tickets": tickets})


@app.route("/api/tickets/<ticket_id>", methods=["GET"])
def get_ticket_by_id(ticket_id):
    result = supabase.table("tickets").select("*").eq("ticket_id", ticket_id).execute()
    return jsonify(result.data[0]) if result.data else (jsonify({"error": "Ticket not found"}), 404)


@app.route("/logs", methods=["GET"])
def view_logs():
    return get_tickets()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "WMS WhatsApp Ticket Logging server is live",
        "endpoints": {
            "/webhook": "POST - incoming messages",
            "/api/tickets": "GET - all tickets (filters: ?warehouse=&status=&priority=)",
            "/api/tickets/<ticket_id>": "GET - single ticket details",
        },
    })


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)