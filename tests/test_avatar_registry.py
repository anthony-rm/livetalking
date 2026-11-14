#!/usr/bin/env python3
"""Unit tests for the avatar generator registry and request validation.

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m tests.test_avatar_registry
"""
import json
import unittest

from server import avatar_routes as ar
from server import enhance_routes as er


class AvatarRegistryTests(unittest.TestCase):
    def test_registry_lists_both_generators(self):
        ids = [m["id"] for m in ar.avatar_generators()]
        self.assertEqual(ids, ["wav2lip", "musetalk", "ditto"])
        self.assertEqual(ar.DEFAULT_GENERATOR, "wav2lip")

    def test_every_param_declares_type_and_default(self):
        for spec in ar.AVATAR_GENERATORS.values():
            for p in spec["params"]:
                self.assertIn(p["type"], ("int", "bool", "select", "text"), p)
                self.assertIn("default", p, p)
                if p["type"] == "select":
                    self.assertIn(p["default"], p["options"], p)

    def test_wav2lip_defaults_match_the_generator_signature(self):
        params = ar.build_task_params("wav2lip", {}, "/tmp/x.mp4")
        self.assertEqual(params["img_size"], 256)
        self.assertEqual(params["pads"], [0, 10, 0, 0])
        self.assertEqual(params["face_det_batch_size"], 16)
        self.assertFalse(params["nosmooth"])

    def test_musetalk_defaults_match_the_generator_signature(self):
        params = ar.build_task_params("musetalk", {}, "/tmp/x.mp4")
        self.assertEqual(params["bbox_shift"], 0)
        self.assertEqual(params["extra_margin"], 10)
        self.assertEqual(params["parsing_mode"], "jaw")
        self.assertEqual(params["version"], "v15")

    def test_form_strings_are_coerced(self):
        params = ar.build_task_params(
            "wav2lip",
            {"img_size": "160", "pads": "1 2 3 4", "nosmooth": "true"}, "/tmp/x.mp4")
        self.assertEqual(params["img_size"], 160)
        self.assertEqual(params["pads"], [1, 2, 3, 4])
        self.assertTrue(params["nosmooth"])

    def test_bad_values_raise(self):
        with self.assertRaises(ValueError):
            ar.build_task_params("wav2lip", {"pads": "0 10 0"}, "/tmp/x.mp4")
        with self.assertRaises(ValueError):
            ar.build_task_params("wav2lip", {"img_size": "big"}, "/tmp/x.mp4")
        with self.assertRaises(ValueError):
            ar.build_task_params("musetalk", {"version": "v99"}, "/tmp/x.mp4")
        with self.assertRaises(ValueError):
            ar.build_task_params("nope", {}, "/tmp/x.mp4")

    def test_unknown_model_is_rejected_by_the_shared_enqueue(self):
        with self.assertRaises(ValueError):
            ar.enqueue_avatar_task("nope", "a", {}, "/tmp/x.mp4")


class EnhanceAvatarChainSpecTests(unittest.TestCase):
    """_parse_avatar_spec is the boundary between the studio form and the chain."""

    def test_nested_json_spec(self):
        spec = er._parse_avatar_spec(
            {"avatar": {"model": "musetalk", "avatar_id": "a1", "params": {"bbox_shift": 3}}})
        self.assertEqual(spec["model"], "musetalk")
        self.assertEqual(spec["avatar_id"], "a1")
        self.assertEqual(spec["params"], {"bbox_shift": 3})

    def test_flat_multipart_fields(self):
        spec = er._parse_avatar_spec({
            "avatar_model": "wav2lip", "avatar_id": "b1",
            "avatar_params": json.dumps({"img_size": 128}),
        })
        self.assertEqual(spec["model"], "wav2lip")
        self.assertEqual(spec["params"], {"img_size": 128})

    def test_no_avatar_spec(self):
        self.assertEqual(er._parse_avatar_spec({}), {})
        self.assertEqual(er._parse_avatar_spec({"width": "1080"}), {})

    def test_partial_or_invalid_specs_raise(self):
        for bad in (
            {"avatar": {"model": "wav2lip"}},                      # no id
            {"avatar_model": "nope", "avatar_id": "x"},            # unsupported model
            {"avatar_model": "wav2lip", "avatar_id": "x",
             "avatar_params": "not json"},                          # malformed params
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                er._parse_avatar_spec(bad)

    def test_avatar_alone_is_enough(self):
        with self.assertRaises(ValueError):
            er._parse_avatar_spec({"avatar_id": "x"})  # an id without a model


if __name__ == "__main__":
    unittest.main(verbosity=2)
