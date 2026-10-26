"""ragaudit —— 自检测试。

沿检索、拼接、生成三段分别体检，定位答案错在哪个环节。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ragaudit  # noqa: E402


class TestRagAudit(unittest.TestCase):
    def test_retrieval_failure_is_attributed(self):
        rows = [{"retrieved": [], "hit_coverage": 0.0, "answer": "无"} for _ in range(10)]
        r = ragaudit.audit(rows)
        retrieval = [s for s in r["stages"] if s["stage"] == "retrieval"][0]
        self.assertTrue(retrieval["findings"])
        self.assertEqual(r["primary_stage"], "retrieval")

    def test_generation_failure_is_attributed(self):
        rows = [{"retrieved": ["a"], "hit_coverage": 0.9,
                 "answer": "x", "has_support": False} for _ in range(10)]
        r = ragaudit.audit(rows)
        generation = [s for s in r["stages"] if s["stage"] == "generation"][0]
        self.assertTrue(generation["findings"])
        self.assertEqual(r["primary_stage"], "generation")

    def test_healthy_pipeline_has_no_issue(self):
        rows = [{"retrieved": ["a", "b"], "hit_coverage": 0.9,
                 "answer": "一个有依据的答案，长度足够" * 3,
                 "has_support": True} for _ in range(20)]
        r = ragaudit.audit(rows)
        self.assertFalse(r["has_issue"])

    def test_truncation_is_detected(self):
        rows = [{"retrieved": ["a"], "hit_coverage": 0.9, "truncated": True,
                 "answer": "有依据的答案" * 5, "has_support": True}
                for _ in range(10)]
        r = ragaudit.audit(rows)
        context = [s for s in r["stages"] if s["stage"] == "context"][0]
        self.assertTrue(context["findings"])

    def test_advice_is_given_per_stage(self):
        rows = [{"retrieved": [], "answer": "无"} for _ in range(5)]
        r = ragaudit.audit(rows)
        for s in r["stages"]:
            self.assertTrue(s["advice"])
