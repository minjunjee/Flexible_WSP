"""Instructions and their checkers.

Ten instructions use the wording and the pass rule of IFEval (google-research/instruction_following_eval,
instructions.py); the rule is re-implemented here in a few lines each and the IFEval id is given. Three are ours:
the JSON instruction of experiment 1, "start with a fixed phrase" and "at most three words" (PLAN.md, experiments 5-6).

check(text) -> bool takes the decoded answer without the end token.
"""
import json
import re

from wsp.score import parse_json

END_PHRASE = "Is there anything else I can help with?"
START_PHRASE = "Sure, here is the answer:"


def _json_format(t):
    v = t.strip()
    for p in ("```json", "```Json", "```JSON", "```"):
        if v.startswith(p):
            v = v[len(p):]
            break
    if v.endswith("```"):
        v = v[:-3]
    try:
        json.loads(v.strip())
        return True
    except ValueError:
        return False


def _french(t):
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    try:
        return detect(t) == "fr"
    except Exception:
        return False


def _bullets(t):
    return len(re.findall(r"^\s*\*[^\*].*$", t, flags=re.MULTILINE)) + len(re.findall(r"^\s*-.*$", t, flags=re.MULTILINE)) == 3


def _title(t):
    return any(x.lstrip("<").rstrip(">").strip() for x in re.findall(r"<<[^\n]+>>", t))


TASKS = {
    "json_answer": dict(
        text='Respond only with a JSON object of the form {"answer": "..."} and nothing else.', source="ours (experiment 1)",
        check=lambda t: parse_json(t)[2]),
    "json_format": dict(
        text="Entire output should be wrapped in JSON format. You can use markdown ticks such as ```.", source="IFEval detectable_format:json_format",
        check=_json_format),
    "lowercase": dict(
        text="Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
        source="IFEval change_case:english_lowercase (without its language test)", check=lambda t: t.islower()),
    "uppercase": dict(
        text="Your entire response should be in English, and in all capital letters.",
        source="IFEval change_case:english_capital (without its language test)", check=lambda t: t.isupper()),
    "title": dict(
        text="Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.", source="IFEval detectable_format:title",
        check=_title),
    "three_bullets": dict(
        text="Your answer must contain exactly 3 bullet points. Use the markdown bullet points such as:\n* This is point 1. \n* This is point 2",
        source="IFEval detectable_format:number_bullet_lists", check=_bullets),
    "end_phrase": dict(
        text=f'Finish your response with this exact phrase "{END_PHRASE}". No other words should follow this phrase.', source="IFEval startend:end_checker",
        check=lambda t: t.strip().strip('"').lower().endswith(END_PHRASE.lower())),
    "quotation": dict(
        text="Wrap your entire response with double quotation marks.", source="IFEval startend:quotation",
        check=lambda t: len(t.strip()) > 1 and t.strip()[0] == '"' and t.strip()[-1] == '"'),
    "postscript": dict(
        text="At the end of your response, please explicitly add a postscript starting with P.S.", source="IFEval detectable_content:postscript",
        check=lambda t: bool(re.findall(r"\s*p\.\s?s\..*$", t.lower(), flags=re.MULTILINE))),
    "no_comma": dict(
        text="In your entire response, refrain from the use of any commas.", source="IFEval punctuation:no_comma",
        check=lambda t: len(t.strip()) > 0 and "," not in t),
    "french": dict(
        text="Your ENTIRE response should be in French language, no other language is allowed.", source="IFEval language:response_language",
        check=_french),
    "start_phrase": dict(
        text=f'Start your response with the exact phrase "{START_PHRASE}".', source="ours (PLAN.md experiment 6)",
        check=lambda t: t.strip().startswith(START_PHRASE)),
    "three_words": dict(
        text="Answer in at most three words.", source="ours (PLAN.md experiment 5)",
        check=lambda t: 0 < len(re.findall(r"[^\W_]+", t)) <= 3),
}
