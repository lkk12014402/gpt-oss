import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Callable

import jinja2
import numpy as np
from tqdm import tqdm
import requests
import io
from .types import EvalResult, Message, SingleEvalResult


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = 128,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap_unordered(f, xs), total=len(xs)))


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )



QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]
def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)

def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )

def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def url_to_fileobj(url: str, binary=False) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content) if binary else io.StringIO(response.text)


def has_only_user_assistant_messages(messages: list[Message]) -> bool:
    """
    Check if the messages only contain user and assistant messages.
    """
    return all(m["role"] in ("user", "assistant") for m in messages)
