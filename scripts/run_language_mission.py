#!/usr/bin/env python3
"""Japanese mission dialogue with clarification, persistence and Gazebo execution."""
import argparse
from datetime import datetime
import json
import locale
import os
import queue
import sys
import threading
import time
from pathlib import Path
from mission_dialogue import Dialogue, ROOT
from mission_language_backend import OpenAIBackend, RuleBackend, BackendError
from mission_ollama_backend import OllamaBackend


def configure_terminal_input():
    """Use Unicode-aware editing instead of the terminal's byte-wise erase."""
    if not sys.stdin.isatty():
        return
    # Python's UTF-8 mode can coexist with libc's POSIX locale. Readline uses
    # libc for multibyte character widths, so fix LC_CTYPE as well.
    if hasattr(locale, "nl_langinfo"):
        encoding = locale.nl_langinfo(locale.CODESET).lower().replace("-", "")
        if encoding != "utf8":
            for name in ["C.UTF-8", "C.utf8"]:
                try:
                    locale.setlocale(locale.LC_CTYPE, name)
                    # Readline initializes its locale again from the environment.
                    os.environ["LC_CTYPE"] = name
                    if os.environ.get("LC_ALL"):
                        os.environ["LC_ALL"] = name
                    break
                except locale.Error:
                    continue
    try:
        import readline
    except ImportError:
        return
    if "libedit" not in (readline.__doc__ or ""):
        readline.parse_and_bind("set input-meta on")
        readline.parse_and_bind("set convert-meta off")
        readline.parse_and_bind("set output-meta on")
        readline.parse_and_bind(r'"\C-h": backward-delete-char')
        readline.parse_and_bind(r'"\C-?": backward-delete-char')


def load_environment(path):
    if not path.exists():
        return
    allowed = {"OPENAI_API_KEY", "OPENAI_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL"}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ['"', "'"]:
                value = value[1:-1]
            if value:
                os.environ.setdefault(key, value)


def make_backend(name, model=None, base_url=None):
    if name == "ollama":
        return OllamaBackend(model, base_url)
    if name == "openai":
        return OpenAIBackend(model)
    if name == "rules":
        return RuleBackend()
    raise ValueError("未対応の解釈方式")


def extract_with_progress(backend, text, context, interval=10):
    """Keep the terminal responsive while a local HTTP request loads a model."""
    result = queue.Queue(maxsize=1)

    def work():
        try:
            result.put((True, backend.extract(text, context)))
        except BaseException as exc:
            result.put((False, exc))

    print(
        "ローカルLLMの準備・解釈中… 初回は数分かかる場合があります。中止はCtrl+C。",
        flush=True,
    )
    started = time.monotonic()
    threading.Thread(target=work, daemon=True).start()
    while True:
        try:
            ok, value = result.get(timeout=interval)
        except queue.Empty:
            print(
                f"  応答待ち：{int(time.monotonic()-started)}秒経過（モデル読込／解釈中）",
                flush=True,
            )
            continue
        if not ok:
            raise value
        print(f"解釈完了（{time.monotonic()-started:.1f}秒）", flush=True)
        return value


def run_dialogue(dialogue, backend, utterances, session_path, interactive=False):
    if interactive:
        configure_terminal_input()
    iterator = iter(utterances)
    retry_required = False
    while True:
        try:
            text = next(iterator)
        except StopIteration:
            if dialogue.ready and not retry_required:
                return "ready"
            initial = not dialogue.turns and not any(
                value is not None for value in dialogue.slots.values()
            )
            if initial:
                print(
                    "運搬指示を一文で入力してください。例：部品置き場から検査場まで背の高い工具箱を慎重に運んで。",
                    flush=True,
                )
            else:
                for q in dialogue.questions():
                    print("確認: " + q["text"].strip(), flush=True)
                if retry_required and dialogue.ready:
                    print(
                        "更新を確認できていません。指示を言い直すか、/quitで終了してください。",
                        flush=True,
                    )
            if not interactive:
                return "needs_clarification"
            try:
                text = input("指示 > " if initial else "回答 > ")
            except (EOFError, KeyboardInterrupt):
                return "cancelled"
        text = text.strip()
        if text in ["/quit", "中止", "キャンセル", "終了"]:
            return "cancelled"
        if text == "/show":
            print(json.dumps(dialogue.slots, ensure_ascii=False, indent=2))
            continue
        if text == "/reset":
            fresh = Dialogue(dialogue.config)
            dialogue.slots = fresh.slots
            dialogue.evidence = {}
            dialogue.issues = {}
            dialogue.turns = []
            retry_required = False
            dialogue.save(session_path)
            continue
        if not text:
            continue
        print("入力: " + text, flush=True)
        try:
            patch = (
                extract_with_progress(backend, text, dialogue.context())
                if backend.name == "ollama"
                else backend.extract(text, dialogue.context())
            )
            dialogue.apply(text, patch, backend.name)
            retry_required = False
        except KeyboardInterrupt:
            print("\n中止しました。会話を保存します。", flush=True)
            dialogue.save(session_path)
            return "cancelled"
        except (BackendError, ValueError) as exc:
            print("解釈エラー: " + str(exc), flush=True)
            dialogue.save(session_path)
            if not interactive:
                return "interpretation_error"
            retry_required = True
            continue
        dialogue.save(session_path)
    return "ready"


def describe(mission):
    s = mission["request"]
    profile = mission["controller_profile"]
    physical = mission["physical_simulation"]
    print("\n運搬条件:", flush=True)
    print(f'  {s["cargo_description"]}: {mission["route"][0]} → {mission["route"][1]}')
    print(
        f'  方針: {profile["label"]}（速度上限 {profile["speed_mps"]:.2f}m/s、加速度上限 {profile["acceleration_mps2"]:.2f}m/s²）'
    )
    print(
        "  シミュレーション寸法: "
        + " × ".join(f"{x*100:g}" for x in physical["size_m"])
        + " cm"
    )
    print(f'  シミュレーション質量: {physical["mass_kg"]:g} kg')
    for note in mission["notes"]:
        print("  ・" + note)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend", choices=["ollama", "openai", "rules"], default="ollama"
    )
    ap.add_argument(
        "--ollama-url", help="Local Ollama URL; defaults to cargo-ollama in Docker"
    )
    ap.add_argument("--model")
    ap.add_argument("--request")
    ap.add_argument("--answers", type=Path, help="JSON array of further user replies")
    ap.add_argument("--resume", type=Path, help="saved session.json")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    ap.add_argument(
        "--sim-cargo", choices=["standard", "slippery", "stable"], default="standard"
    )
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--domain-id", type=int, default=95)
    args = ap.parse_args()
    if not 1 <= args.domain_id <= 100:
        ap.error("domain-id must be 1..100")
    load_environment(args.env_file)
    try:
        backend = make_backend(args.backend, args.model, args.ollama_url)
    except BackendError as exc:
        print(str(exc))
        return 2
    dialogue = Dialogue.restore(args.resume) if args.resume else Dialogue()
    folder = args.output or ROOT / "logs/language_mission" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    folder.mkdir(parents=True, exist_ok=False)
    session = folder / "session.json"
    dialogue.save(session)
    print(
        "解釈方式: "
        + backend.name
        + (
            "（オフライン動作確認用。自由な言い換えはLLM接続時に対応）"
            if args.backend == "rules"
            else " / " + backend.model
        ),
        flush=True,
    )
    print("会話保存先: " + str(session), flush=True)
    replies = []
    if args.request:
        replies.append(args.request)
    if args.answers:
        answers = json.loads(args.answers.read_text(encoding="utf-8-sig"))
        if not isinstance(answers, list) or any(
            not isinstance(x, str) for x in answers
        ):
            raise ValueError("回答ファイルは文字列のJSON配列にしてください")
        replies.extend(answers)
    if not replies and not args.resume:
        print("/showで現在の条件、/quitで保存終了できます。")
    try:
        state = run_dialogue(
            dialogue, backend, replies, session, interactive=args.answers is None
        )
    finally:
        if isinstance(backend, OllamaBackend):
            print(
                "ローカルモデルのメモリ解放: "
                + (
                    "完了"
                    if backend.release()
                    else "未確認（サーバー側で自動解放を待ちます）"
                ),
                flush=True,
            )
    (folder / "dialogue_result.json").write_text(
        json.dumps(
            dict(
                status=state,
                backend=backend.name,
                model=getattr(backend, "model", None),
                questions=dialogue.questions(),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if state != "ready":
        return 0 if state == "cancelled" else 2
    mission = dialogue.compile(args.sim_cargo)
    describe(mission)
    # ROS is imported by the runner only during execution, after all questions.
    import mission_execution

    run_folder = folder / "run"
    c, route = mission_execution.prepare(mission, run_folder)
    print(f'経路: {route["length_m"]:.2f}m / 記録: {run_folder}', flush=True)
    if args.plan_only:
        print("経路とworldを保存しました。Gazeboは起動していません。")
        return 0
    return mission_execution.execute(
        mission, run_folder, c, route, args.gui, args.domain_id
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("入力・保存エラー: " + str(exc))
        raise SystemExit(2)
