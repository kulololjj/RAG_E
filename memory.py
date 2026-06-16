"""
会话记忆模块 — 持久化 + 跨会话 + 多轮
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

MEMORY_DIR = Path(__file__).parent / "conversations"


def _ensure_dir():
    os.makedirs(MEMORY_DIR, exist_ok=True)


def list_sessions() -> List[Dict]:
    """列出所有已保存的会话"""
    _ensure_dir()
    sessions = []
    for f in sorted(MEMORY_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "id": f.stem,
                "title": data.get("title", f.stem),
                "messages": len(data.get("history", [])),
                "created": data.get("created", ""),
                "updated": data.get("updated", ""),
            })
        except Exception:
            pass
    return sessions


def load_session(session_id: str) -> Optional[Dict]:
    """加载指定会话"""
    path = MEMORY_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(session_id: str, history: List[str], messages: List[Dict],
                 title: str = "新对话"):
    """保存会话到磁盘"""
    _ensure_dir()
    data = {
        "id": session_id,
        "title": title,
        "history": history,
        "messages": messages,
        "created": load_session(session_id).get("created", datetime.now().isoformat()) if (MEMORY_DIR / f"{session_id}.json").exists() else datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
    }
    (MEMORY_DIR / f"{session_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_session(session_id: str):
    """删除指定会话"""
    path = MEMORY_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()


def new_session_id() -> str:
    """生成新会话 ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_chat_prompt(history: List[str], max_exchanges: int = 3) -> str:
    """
    将对话历史格式化为 LLM prompt 片段（最近 N 轮）。
    history 格式: ["用户: xxx", "AI: xxx", ...]
    """
    if not history:
        return ""
    recent = history[-(max_exchanges * 2):]
    return "\n".join(recent)
