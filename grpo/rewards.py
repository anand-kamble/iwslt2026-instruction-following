"""Task-routed GRPO reward functions for IWSLT 2026 Stage 4 (text-only).

Task routing follows the ms-swift `GRPO/DeveloperGuide/multi_task.md` pattern:
each reward ORM returns `None` for rows whose `task` column is outside its
family, and ms-swift aggregates only non-None values per row.

Register with:
    swift rlhf \\
        --rlhf_type grpo \\
        --external_plugins grpo/rewards.py \\
        --reward_funcs mt_comet t1_jaccard t5_numeric prometheus_judge \\
                       language_match repetition soft_overlong \\
        --reward_weights 1.0 1.0 1.0 1.0 0.3 0.2 0.2

Expected dataset columns: `messages` (system+user), `solution` (gold), `task`.
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from typing import Any, List, Optional

try:
    from swift.plugin import ORM, orms
except ImportError:
    from swift.rewards import ORM, orms


# -----------------------------------------------------------------------------
# Task families
# -----------------------------------------------------------------------------
MT_TASKS = {"mt_de", "mt_it", "mt_zh", "s2tt_de", "s2tt_it", "s2tt_zh"}
QA_TASKS = {
    "qa_en", "qa_de", "qa_it", "qa_zh",
    "sqa_en", "sqa_de", "sqa_it", "sqa_zh",
}
JUDGE_SYNTH_TASKS = {
    "T3_gist_en", "T4_topic",
    "T6_gist_de", "T7_gist_zh",
    "T8_tone_en", "T9_tone_de", "T10_tone_zh",
}
JUDGE_TASKS = JUDGE_SYNTH_TASKS | QA_TASKS
T1_TASK = "T1_keywords"
T5_TASK = "T5_numeric_qa"

# Tasks that have a required output language (drives language_match)
LANG_LOCKED_TASKS = {
    "mt_de": "de", "mt_it": "it", "mt_zh": "zh",
    "s2tt_de": "de", "s2tt_it": "it", "s2tt_zh": "zh",
    "T6_gist_de": "de", "T7_gist_zh": "zh",
    "T9_tone_de": "de", "T10_tone_zh": "zh",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _strip_answer_prefix(text: str) -> str:
    """Drop any 'Your answer:' / 'Answer:' prefix the model re-emits."""
    return re.sub(r"^\s*(Your\s+answer|Answer)\s*:\s*", "", text or "", flags=re.IGNORECASE).strip()


def _normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles/punct."""
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =============================================================================
# 1. MT — COMET-DA (CPU, lazy-loaded)
# =============================================================================
class MTCometReward(ORM):
    """COMET-DA sentence score in [0, 1] for MT / S2TT.

    Uses Unbabel/wmt22-comet-da via the `comet` library, pinned to CPU so the
    policy keeps the training GPU. Lazy-loaded on first call.

    Returns None for non-MT rows.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from comet import download_model, load_from_checkpoint
            ckpt = download_model("Unbabel/wmt22-comet-da")
            self._model = load_from_checkpoint(ckpt)
            try:
                self._model = self._model.to("cpu")
            except Exception:
                pass
        return self._model

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        task: Optional[List[str]] = None,
        messages: Optional[List[List[dict]]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        n = len(completions)
        solution = solution or [""] * n
        task = task or [""] * n
        messages = messages or [None] * n

        indices, triples = [], []
        for i, (hyp, ref, t, msgs) in enumerate(zip(completions, solution, task, messages)):
            if t not in MT_TASKS:
                continue
            hyp = _strip_answer_prefix(hyp)
            ref = (ref or "").strip()
            src = _extract_source_text(msgs) or ""
            if not hyp or not ref:
                continue
            indices.append(i)
            triples.append({"src": src, "mt": hyp, "ref": ref})

        rewards: List[Optional[float]] = [None] * n
        if not triples:
            # still mark in-family rows with 0 when completion/ref is empty
            for i, t in enumerate(task):
                if t in MT_TASKS and rewards[i] is None:
                    rewards[i] = 0.0
            return rewards

        try:
            scores = self._get_model().predict(triples, batch_size=8, gpus=0).scores
        except Exception as exc:
            print(f"[mt_comet] prediction failed: {exc!r}; returning zeros")
            scores = [0.0] * len(triples)

        for idx, s in zip(indices, scores):
            rewards[idx] = _clip01(float(s))
        for i, t in enumerate(task):
            if t in MT_TASKS and rewards[i] is None:
                rewards[i] = 0.0
        return rewards


def _extract_source_text(msgs) -> Optional[str]:
    """Pull English source text from '<text>...</text>' in the user turn."""
    if not msgs:
        return None
    for m in msgs:
        if m.get("role") == "user":
            content = m.get("content", "") or ""
            m2 = re.search(r"<text>(.*?)</text>", content, flags=re.DOTALL)
            if m2:
                return m2.group(1).strip()
    return None


# =============================================================================
# 2. MT chrF (retained as a cheap fallback / smoke-test reward)
# =============================================================================
class MTChrFReward(ORM):
    """Sentence chrF in [0, 1] for MT / S2TT. Fast; good for smoke tests."""

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        task: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        import sacrebleu
        n = len(completions)
        solution = solution or [""] * n
        task = task or [""] * n
        rewards: List[Optional[float]] = []
        for hyp, ref, t in zip(completions, solution, task):
            if t not in MT_TASKS:
                rewards.append(None); continue
            hyp = _strip_answer_prefix(hyp)
            ref = (ref or "").strip()
            if not hyp or not ref:
                rewards.append(0.0); continue
            try:
                rewards.append(_clip01(sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0))
            except Exception:
                rewards.append(0.0)
        return rewards


# =============================================================================
# 3. QA token-F1 (retained as a cheap fallback for smoke tests)
# =============================================================================
class QATokenF1Reward(ORM):
    UNANSWERABLE_PATTERNS = (
        "not answerable", "unanswerable",
        "nicht zu beantworten", "non è possibile rispondere", "无法回答",
    )

    def _is_unanswerable(self, s: str) -> bool:
        s = (s or "").lower().strip()
        return any(p in s for p in self.UNANSWERABLE_PATTERNS)

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        task: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        n = len(completions)
        solution = solution or [""] * n
        task = task or [""] * n
        rewards: List[Optional[float]] = []
        for hyp, ref, t in zip(completions, solution, task):
            if t not in QA_TASKS:
                rewards.append(None); continue
            hyp = _strip_answer_prefix(hyp)
            ref = (ref or "").strip()
            if not ref:
                rewards.append(0.0); continue
            ru, hu = self._is_unanswerable(ref), self._is_unanswerable(hyp)
            if ru or hu:
                rewards.append(1.0 if (ru and hu) else 0.0); continue
            ht, rt = _normalize_answer(hyp).split(), _normalize_answer(ref).split()
            if not ht or not rt:
                rewards.append(0.0); continue
            common = sum((Counter(ht) & Counter(rt)).values())
            if common == 0:
                rewards.append(0.0); continue
            p, r = common / len(ht), common / len(rt)
            rewards.append(2 * p * r / (p + r))
        return rewards


# =============================================================================
# 4. T1 — Jaccard on comma-split keyword sets
# =============================================================================
class T1JaccardReward(ORM):
    """Jaccard similarity between comma-split keyword sets (case-insensitive)."""

    def _tokenize(self, s: str) -> set:
        return {kw.strip().lower() for kw in (s or "").split(",") if kw.strip()}

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        task: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        n = len(completions)
        solution = solution or [""] * n
        task = task or [""] * n
        rewards: List[Optional[float]] = []
        for hyp, ref, t in zip(completions, solution, task):
            if t != T1_TASK:
                rewards.append(None); continue
            hyp_set = self._tokenize(_strip_answer_prefix(hyp))
            ref_set = self._tokenize(ref)
            if not ref_set:
                rewards.append(0.0); continue
            if not hyp_set:
                rewards.append(0.0); continue
            inter = hyp_set & ref_set
            union = hyp_set | ref_set
            rewards.append(len(inter) / max(1, len(union)))
        return rewards


# =============================================================================
# 5. T5 — Numeric-answer match
# =============================================================================
class T5NumericReward(ORM):
    """Extract the first number from hyp/ref and compare; 1 on exact match else 0."""

    NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

    def _first_number(self, s: str) -> Optional[float]:
        if not s:
            return None
        m = self.NUM_RE.search(s.replace(",", ""))
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        task: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        n = len(completions)
        solution = solution or [""] * n
        task = task or [""] * n
        rewards: List[Optional[float]] = []
        for hyp, ref, t in zip(completions, solution, task):
            if t != T5_TASK:
                rewards.append(None); continue
            h = self._first_number(_strip_answer_prefix(hyp))
            r = self._first_number(ref)
            if r is None:
                rewards.append(0.0); continue
            if h is None:
                rewards.append(0.0); continue
            rewards.append(1.0 if abs(h - r) < 1e-6 else 0.0)
        return rewards


# =============================================================================
# 6. Language match (heuristic, CPU-only)
# =============================================================================
class LanguageMatchReward(ORM):
    """1.0 if the completion appears to be in the target language, else 0.0.
    Returns None for tasks without a locked output language.
    """

    CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    DE_CHARS = set("äöüÄÖÜß")
    IT_CHARS = set("àèéìíîòóùú")

    def _detect(self, text: str) -> str:
        if self.CJK_RE.search(text):
            return "zh"
        if any(c in self.DE_CHARS for c in text):
            return "de"
        if any(c in self.IT_CHARS for c in text):
            return "it"
        lower = text.lower()
        if re.search(r"\b(der|die|das|und|ist|nicht|ich|sie)\b", lower):
            return "de"
        if re.search(r"\b(il|la|le|e|è|non|che|sono|questa)\b", lower):
            return "it"
        return "en"

    def __call__(
        self,
        completions: List[str],
        task: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        n = len(completions)
        task = task or [""] * n
        rewards: List[Optional[float]] = []
        for hyp, t in zip(completions, task):
            target = LANG_LOCKED_TASKS.get(t)
            if target is None:
                rewards.append(None); continue
            detected = self._detect(_strip_answer_prefix(hyp))
            rewards.append(1.0 if detected == target else 0.0)
        return rewards


# =============================================================================
# 7. Prometheus judge (on-cluster via swift deploy OpenAI endpoint)
# =============================================================================
PROMETHEUS_RUBRICS = {
    "gist": (
        "Score whether the response captures the main claim of the source as a "
        "single concise sentence without enumeration.\n"
        "Score 1: Wrong claim or filler summary.\n"
        "Score 2: Multiple sentences or missed claim, partial coverage only.\n"
        "Score 3: Correct claim but runs long or mildly enumerative.\n"
        "Score 4: One sentence, correct claim, slightly wordy.\n"
        "Score 5: One sentence, the single most central claim, names and numbers preserved."
    ),
    "tone": (
        "Score whether the response describes the speaker's tone, pace, and delivery "
        "style in 1-2 sentences in the target language.\n"
        "Score 1: Contradicts the reference or empty.\n"
        "Score 2: Vague OR wrong target language.\n"
        "Score 3: Partially accurate OR 1 sentence is fine but slightly off.\n"
        "Score 4: Accurate but generic description.\n"
        "Score 5: Accurate and specific description, target language, 1-2 sentences."
    ),
    "extraction": (
        "Score whether the response returns the requested format and items that "
        "match the reference.\n"
        "Score 1: Empty, invalid format, or unrelated items.\n"
        "Score 2: Wrong format OR only one or two correct items.\n"
        "Score 3: Correct format but significant recall gap.\n"
        "Score 4: Correct format, minor recall gap OR one hallucinated item.\n"
        "Score 5: Correct format, full recall on reference items, no hallucinated items."
    ),
    "qa": (
        "Score whether the response correctly answers the question with information "
        "from the source.\n"
        "Score 1: Wrong answer or unrelated.\n"
        "Score 2: Partial answer, important facts missed.\n"
        "Score 3: Roughly correct but imprecise or incomplete.\n"
        "Score 4: Correct, minor wording or brevity issues.\n"
        "Score 5: Correct, precise, directly answers the question."
    ),
}

TASK_TO_RUBRIC = {
    "T3_gist_en":  "gist",  "T6_gist_de":  "gist", "T7_gist_zh": "gist",
    "T4_topic":    "extraction",
    "T8_tone_en":  "tone",  "T9_tone_de":  "tone", "T10_tone_zh": "tone",
    **{t: "qa" for t in QA_TASKS},
}

PROM_SYSTEM_MESSAGE = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute "
    "standards set for performance."
)

SCORE_MAPPING = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}


def _build_prometheus_prompt(instruction: str, response: str, reference: str, rubric: str) -> str:
    return (
        "###Task Description:\n"
        "An instruction, a response to evaluate, a reference answer that gets a score of 5, "
        "and a score rubric representing an evaluation criteria are given.\n"
        "1. Write detailed feedback on the response, strictly by the rubric.\n"
        "2. Then write a score integer between 1 and 5.\n"
        '3. Output format: "Feedback: (write feedback) [RESULT] (integer 1-5)"\n'
        "4. Do not generate any other opening, closing, or explanations.\n\n"
        f"###The instruction to evaluate:\n{instruction}\n"
        f"###Response to evaluate:\n{response}\n\n"
        f"###Reference Answer:\n{reference}\n\n"
        "###Score Rubrics:\n"
        f"{rubric}\n\n"
        "###Feedback:\n[RESULT]"
    )


class PrometheusJudgeReward(ORM):
    """Rubric-graded reward via a co-deployed Prometheus-7B server.

    The server is expected to speak the OpenAI chat-completions API, hosted at
    PROMETHEUS_URL (default http://127.0.0.1:8000/v1). Launch on a side GPU
    with: `swift deploy --model prometheus-eval/prometheus-7b-v2.0 --port 8000`.

    Returns a scalar in [0, 1] for judge-eligible tasks, None otherwise. If the
    server is unreachable, returns None so ms-swift aggregates without the judge
    rather than zeroing out the reward.
    """

    FINAL_RESULT_RE = re.compile(r"\[?\s*RESULT\s*\]?\s*[:\-]?\s*([1-5])\b", re.IGNORECASE)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._client = None
        self._model_id = None
        self._unavailable_logged = False

    def _get_client(self):
        if self._client is not None:
            return self._client, self._model_id
        try:
            from openai import OpenAI
        except ImportError:
            if not self._unavailable_logged:
                print("[prometheus_judge] openai package not installed; skipping judge.")
                self._unavailable_logged = True
            return None, None
        base_url = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:8000/v1")
        api_key = os.environ.get("PROMETHEUS_API_KEY", "EMPTY")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list().data
            model_id = models[0].id if models else "prometheus"
        except Exception as exc:
            if not self._unavailable_logged:
                print(f"[prometheus_judge] server at {base_url} unreachable: {exc!r}")
                self._unavailable_logged = True
            return None, None
        self._client = client
        self._model_id = model_id
        return client, model_id

    def _score(self, client, model_id, user_content: str) -> Optional[float]:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": PROM_SYSTEM_MESSAGE},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[prometheus_judge] request failed: {exc!r}")
            return None
        m = self.FINAL_RESULT_RE.search(text)
        if not m:
            return None
        score = max(1, min(5, int(m.group(1))))
        return SCORE_MAPPING[score]

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        task: Optional[List[str]] = None,
        messages: Optional[List[List[dict]]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        n = len(completions)
        solution = solution or [""] * n
        task = task or [""] * n
        messages = messages or [None] * n

        rewards: List[Optional[float]] = [None] * n
        # Pre-filter: which rows actually need the judge?
        needed = [i for i, t in enumerate(task) if t in TASK_TO_RUBRIC]
        if not needed:
            return rewards

        client, model_id = self._get_client()
        if client is None:
            return rewards  # all None — server not up, let other rewards drive

        for i in needed:
            t = task[i]
            rubric_name = TASK_TO_RUBRIC[t]
            rubric = PROMETHEUS_RUBRICS[rubric_name]
            instruction = _instruction_from_messages(messages[i]) or ""
            user_content = _build_prometheus_prompt(
                instruction=instruction,
                response=_strip_answer_prefix(completions[i]),
                reference=(solution[i] or "").strip(),
                rubric=rubric,
            )
            rewards[i] = self._score(client, model_id, user_content)
        return rewards


def _instruction_from_messages(msgs) -> Optional[str]:
    """Pull the user turn's instruction (between 'Question:' and 'Your answer:')."""
    if not msgs:
        return None
    for m in msgs:
        if m.get("role") == "user":
            content = m.get("content", "") or ""
            m2 = re.search(r"Question:\s*(.*?)\s*Your answer:", content, flags=re.DOTALL)
            if m2:
                return m2.group(1).strip()
            return content.strip()
    return None


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
orms["mt_comet"] = MTCometReward
orms["mt_chrf"] = MTChrFReward            # fallback / smoke-test
orms["qa_token_f1"] = QATokenF1Reward     # fallback / smoke-test
orms["t1_jaccard"] = T1JaccardReward
orms["t5_numeric"] = T5NumericReward
orms["language_match"] = LanguageMatchReward
orms["prometheus_judge"] = PrometheusJudgeReward
