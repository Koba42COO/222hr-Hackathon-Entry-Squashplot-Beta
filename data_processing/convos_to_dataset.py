#!/usr/bin/env python3
import os, json, re, argparse
from pathlib import Path

"""
Scans a folder for:
- RMM memory: research_data/rmm_memory.jsonl (USER / AIVA pairs)
- Any .jsonl with "USER:" / "AIVA" or "assistant" / "user"
- Any .txt/.md transcripts that look like chat

Outputs: data/aiva_convos.jsonl in an instruction-tuning format:
{"conversations":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
"""

SYSTEM_DEFAULT = (
  "You are AiVA. Be clear, kind, and grounded. Use φ-balanced reasoning, "
  "cite sources if you rely on external math, and say when you're unsure."
)

def read_jsonl(p: Path):
    for line in p.open():
        line=line.strip()
        if not line: continue
        try:
            yield json.loads(line)
        except Exception:
            continue

def parse_rmm(path: Path):
    # expects lines with meta.kind == "dialogue" or "agent_interaction" and content "USER: ...\nAIVA..." or "USER: ...\nRESPONSE: ..."
    for obj in read_jsonl(path):
        kind = obj.get("meta",{}).get("kind", "")
        if kind not in ["dialogue", "agent_interaction"]:
            continue
        text = obj.get("content","")
        # Try multiple patterns for splitting conversations
        patterns = [
            r'\bAIVA(?:\[.*?\])?:',
            r'\bRESPONSE:',
            r'\bAIVA:',
            r'\bAssistant:'
        ]

        user_text = ""
        assistant_text = ""

        for pattern in patterns:
            m = re.split(pattern, text, maxsplit=1)
            if len(m) == 2:
                user_text = re.sub(r'^USER:\s*','',m[0]).strip()
                assistant_text = m[1].strip()
                break

        # If no pattern matched, try to extract from the whole text
        if not user_text and "USER:" in text:
            # Simple extraction
            parts = text.split("USER:")
            if len(parts) > 1:
                user_part = parts[1].split("\n", 1)[0] if "\n" in parts[1] else parts[1]
                user_text = user_part.strip()
                # Look for response part
                response_patterns = ["RESPONSE:", "AIVA:", "Assistant:"]
                for rp in response_patterns:
                    if rp in text:
                        resp_part = text.split(rp, 1)[1].split("\n", 1)[0] if "\n" in text.split(rp, 1)[1] else text.split(rp, 1)[1]
                        assistant_text = resp_part.strip()
                        break

        if user_text and assistant_text:
            yield {"user": user_text, "assistant": assistant_text}

def parse_plaintext_chat(p: Path):
    # looks for alternating lines like "User: ..." / "Aiva: ..."
    lines = p.read_text(errors="ignore").splitlines()
    pairs=[]
    cur_user=[]
    cur_ass=[]
    turn="user"
    for ln in lines:
        l = ln.strip()
        if re.match(r'^(USER|User|U):', l):
            if cur_user or cur_ass:
                if cur_user and cur_ass:
                    pairs.append({"user":"\n".join(cur_user).strip(),
                                  "assistant":"\n".join(cur_ass).strip()})
                cur_user, cur_ass = [], []
            cur_user.append(re.sub(r'^(USER|User|U):\s*','',l))
            turn="assistant"
        elif re.match(r'^(AIVA|Assistant|AiVA|Aiva|A):', l):
            cur_ass.append(re.sub(r'^(AIVA|Assistant|AiVA|Aiva|A):\s*','',l))
            turn="user"
        else:
            # keep appending to current speaker if any
            if turn=="assistant" and cur_ass:
                cur_ass.append(l)
            elif turn=="user" and cur_user:
                cur_user.append(l)
    if cur_user and cur_ass:
        pairs.append({"user":"\n".join(cur_user).strip(),
                      "assistant":"\n".join(cur_ass).strip()})
    for p in pairs:
        if p["user"] and p["assistant"]:
            yield p

def parse_generic_jsonl(p: Path):
    # tries common formats (ShareGPT / OpenAI logs / chatml)
    for obj in read_jsonl(p):
        if "conversations" in obj:
            msgs = obj["conversations"]
            u=None;a=None;sys=None
            for m in msgs:
                if m.get("role")=="system" and not sys:
                    sys = m.get("content","")
                if m.get("role")=="user": u = m.get("content","")
                if m.get("role") in ("assistant","model"): a = m.get("content","")
            if u and a:
                yield {"user":u,"assistant":a,"system":sys}
        elif "messages" in obj:
            u=None;a=None;sys=None
            for m in obj["messages"]:
                r = m.get("role")
                if r=="system" and not sys: sys = m.get("content","")
                if r=="user": u = m.get("content","")
                if r in ("assistant","model"): a = m.get("content","")
            if u and a:
                yield {"user":u,"assistant":a,"system":sys}
        else:
            # fallback fields
            u = obj.get("user") or obj.get("question")
            a = obj.get("assistant") or obj.get("answer")
            if u and a:
                yield {"user":u, "assistant":a, "system":obj.get("system")}

def main(root: str):
    rootp = Path(root).resolve()
    outp = Path("data"); outp.mkdir(exist_ok=True, parents=True)
    dst = outp / "aiva_convos.jsonl"
    cnt=0
    with dst.open("w") as w:
        # 1) RMM default path
        rmm = rootp / "research_data" / "rmm_memory.jsonl"
        if rmm.exists():
            for pair in parse_rmm(rmm):
                rec={"conversations":[
                    {"role":"system","content":SYSTEM_DEFAULT},
                    {"role":"user","content":pair["user"]},
                    {"role":"assistant","content":pair["assistant"]}
                ]}
                w.write(json.dumps(rec, ensure_ascii=False)+"\n")
                cnt+=1

        # 2) Walk folder for jsonl/txt/md
        for p in rootp.rglob("*"):
            if p.is_dir(): continue
            if p.name=="rmm_memory.jsonl": continue
            try:
                if p.suffix.lower()==".jsonl":
                    for pair in parse_generic_jsonl(p):
                        rec={"conversations":[
                            {"role":"system","content":pair.get("system") or SYSTEM_DEFAULT},
                            {"role":"user","content":pair["user"]},
                            {"role":"assistant","content":pair["assistant"]}
                        ]}
                        w.write(json.dumps(rec, ensure_ascii=False)+"\n"); cnt+=1
                elif p.suffix.lower() in (".txt",".md"):
                    for pair in parse_plaintext_chat(p):
                        rec={"conversations":[
                            {"role":"system","content":SYSTEM_DEFAULT},
                            {"role":"user","content":pair["user"]},
                            {"role":"assistant","content":pair["assistant"]}
                        ]}
                        w.write(json.dumps(rec, ensure_ascii=False)+"\n"); cnt+=1
            except Exception:
                continue
    print(f"✅ Wrote {cnt} conversation pairs → {dst}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Folder containing your convos")
    args = ap.parse_args()
    main(args.root)
