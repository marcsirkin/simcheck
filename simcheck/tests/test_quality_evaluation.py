"""Tests for golden-set evaluation helpers (simcheck.quality.evaluation)."""

from pathlib import Path

import pytest

from simcheck.quality.classifier import Answer, FakeClassifier
from simcheck.quality.evaluation import (
    EvaluationError,
    GoldenLabel,
    answers_from_json,
    answers_to_json,
    band_changes,
    compute_metrics,
    load_golden,
    rate_with_answers,
    read_cache,
    simulate_hybrid,
    slider_to_level,
    write_cache,
)
from simcheck.quality.page_quality import rate_page
from simcheck.quality.rubric import QuestionKind, questions_for
from simcheck.quality.snapshot import parse_snapshot


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def snap():
    html = (FIXTURES / "good_article.html").read_text()
    filler = "<p>" + "DKIM signatures protect message integrity in transit. " * 10 + "</p>"
    return parse_snapshot("https://example.com/dkim", html.replace("</article>", filler + "</article>"))


def _answers(**over):
    return FakeClassifier(over, name="jev").classify({}, questions_for(None))


class TestLabels:
    @pytest.mark.parametrize("label,level", [("Lowest", 0.0), ("Lowest+", 0.5), ("medium", 2.0),
                                             ("Medium+", 2.5), (" High+ ", 3.5), ("Highest", 4.0)])
    def test_slider_to_level(self, label, level):
        assert slider_to_level(label) == level

    def test_unknown_label(self):
        with pytest.raises(EvaluationError):
            slider_to_level("Great")

    def test_load_golden_skips_unrated_and_excluded(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text(
            "url,include,page_quality,ymyl,purpose,trust,notes\n"
            "https://a.com/,yes,High+,no,commercial,High,good\n"
            "https://b.com/,yes,,,,,\n"
            "https://c.com/,no,Low,,,,blocked\n"
            "https://d.com/,YES,Low,Clearly,,,\n")
        labels = load_golden(p)
        assert [g.url for g in labels] == ["https://a.com/", "https://d.com/"]
        assert labels[0].level == 3.5 and labels[0].trust == 3.0 and labels[0].purpose == "commercial"
        assert labels[1].ymyl == "clearly" and labels[1].trust is None

    def test_bad_trust_label(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text("url,include,page_quality,ymyl,purpose,trust,notes\nhttps://a.com/,yes,High,,,Great,\n")
        with pytest.raises(EvaluationError, match="trust"):
            load_golden(p)


class TestCache:
    def test_answers_round_trip(self):
        original = {
            "trust": Answer("trust", QuestionKind.SCORE, 2.6, {2: 0.4, 3: 0.6}, 0.7, "jev"),
            "ymyl": Answer("ymyl", QuestionKind.CHOICE, "no", {"no": 0.8}, 0.6, "jev"),
            "ads_obstruct": Answer("ads_obstruct", QuestionKind.NOUL, 0.1, {}, 0.8, "claude", "No ads."),
        }
        assert answers_from_json(answers_to_json(original)) == original

    def test_unknown_question_rejected(self):
        with pytest.raises(EvaluationError, match="rubric changed"):
            answers_from_json({"old_question": {"kind": "noul", "value": 0.5, "confidence": 0}})

    def test_jsonl_cache(self, tmp_path):
        p = tmp_path / "a.jsonl"
        write_cache(p, {"url": "u", "backend": "jev", "rubric_version": "v", "answers": {}})
        write_cache(p, {"url": "u", "backend": "claude", "rubric_version": "v", "answers": {}})
        assert set(read_cache(p)) == {("u", "jev", "v"), ("u", "claude", "v")}
        assert read_cache(tmp_path / "missing.jsonl") == {}


class TestSimulateHybrid:
    def test_merges_only_escalated(self):
        jev = _answers()
        jev["trust"] = Answer("trust", QuestionKind.SCORE, 2.0, {}, 0.3, "jev")
        claude = FakeClassifier({"trust": 1.0, "page_quality": 0.0}, name="claude").classify({}, questions_for(None))
        merged, escalated = simulate_hybrid(jev, claude, 0.6)
        assert escalated == ["trust"]
        assert merged["trust"].backend == "claude" and merged["page_quality"].backend == "jev"

    def test_threshold_zero_escalates_nothing_confident(self):
        merged, escalated = simulate_hybrid(_answers(), _answers(), 0.0)
        assert escalated == []

    def test_replay_matches_live_rating(self, snap):
        answers = _answers(page_quality=2.0)
        assert rate_with_answers(snap, answers).band == rate_page(snap, FakeClassifier({"page_quality": 2.0})).band


class TestMetrics:
    def _pair(self, snap, human, predicted, **label):
        return (GoldenLabel(url="u", level=human, **{"ymyl": None, "purpose": None, "trust": None, **label}),
                rate_with_answers(snap, _answers(page_quality=predicted)))

    def test_perfect_agreement(self, snap):
        m = compute_metrics([self._pair(snap, 3.0, 3.0), self._pair(snap, 2.0, 2.0)])
        assert m.exact_slider == 1.0 and m.within_one == 1.0 and m.mae_levels == 0.0 and m.mean_bias == 0.0

    def test_bias_and_tolerance(self, snap):
        m = compute_metrics([self._pair(snap, 2.0, 3.0), self._pair(snap, 2.0, 2.5)])
        assert m.exact_slider == 0.0
        assert m.within_half == 0.5
        assert m.within_one == 1.0
        assert m.mean_bias == pytest.approx(0.75)

    def test_secondary_agreement(self, snap):
        m = compute_metrics([self._pair(snap, 3.0, 3.0, ymyl="no", purpose="informational", trust=2.0)])
        assert m.ymyl_agreement == 1.0 and m.purpose_agreement == 1.0
        assert m.trust_mae == pytest.approx(1.0)  # fake trust = 3.0

    def test_unrated_excluded(self, snap):
        thin = parse_snapshot("https://x/", "<p>tiny</p>")
        unrated = (GoldenLabel("x", 1.0, None, None, None), rate_with_answers(thin, _answers()))
        m = compute_metrics([unrated, self._pair(snap, 3.0, 3.0)])
        assert m.n == 1

    def test_all_unrated_raises(self):
        thin = parse_snapshot("https://x/", "<p>tiny</p>")
        with pytest.raises(EvaluationError):
            compute_metrics([(GoldenLabel("x", 1.0, None, None, None), rate_with_answers(thin, _answers()))])

    def test_band_changes(self, snap):
        a = [rate_with_answers(snap, _answers(page_quality=v)) for v in (3.0, 2.0, 1.0)]
        b = [rate_with_answers(snap, _answers(page_quality=v)) for v in (3.0, 3.0, 1.5)]
        c = band_changes(a, b)
        assert c == {"pages": 3, "slider_changed": 2, "level_changed_1plus": 1,
                     "mean_abs_shift": pytest.approx(0.5), "mean_shift": pytest.approx(0.5)}
