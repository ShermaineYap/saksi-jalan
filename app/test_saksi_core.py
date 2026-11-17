"""Unit tests for the parts of Saksi Jalan that don't need the models."""
from PIL import Image

from saksi_core import Verdict, decide, dominant_colour, name_colour, witness_statement


def test_colour_names():
    assert name_colour((250, 250, 250)) == "white"
    assert name_colour((15, 15, 15)) == "black"
    assert name_colour((190, 190, 195)) == "silver"
    assert name_colour((95, 95, 100)) == "grey"
    assert name_colour((200, 20, 20)) == "red"
    assert name_colour((20, 60, 200)) == "blue"
    assert name_colour((230, 200, 20)) == "yellow"
    assert name_colour((40, 160, 60)) == "green"


def test_dominant_colour_of_flat_image():
    assert dominant_colour(Image.new("RGB", (200, 120), (200, 20, 20))) == "red"


def test_decide_confident_and_unsure():
    top3, ok = decide({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}, threshold=0.7)
    assert ok and top3[0] == ("A", 0.9) and len(top3) == 3
    top3, ok = decide({"A": 0.4, "B": 0.35, "C": 0.25}, threshold=0.7)
    assert not ok and [n for n, _ in top3] == ["A", "B", "C"]


def test_statement_confident():
    v = Verdict(top3=[("Perodua Myvi", 0.9), ("Perodua Axia", 0.05), ("Proton Iriz", 0.03)], confident=True,
                colour_en="white", colour_ms="putih")
    en, ms = witness_statement(v, when="17 Nov 2026, 8 pm", where="Jalan Ampang", plate="wa 12")
    assert "a white Perodua Myvi" in en and "WA 12" in en
    assert "Perodua Myvi berwarna putih" in ms and "WA 12" in ms
    assert "not proof" in en and "bukan bukti" in ms


def test_statement_unsure_lists_candidates_and_never_invents():
    v = Verdict(top3=[("Perodua Myvi", 0.4), ("Perodua Axia", 0.3), ("Proton Iriz", 0.2)], confident=False,
                colour_en="red", colour_ms="merah")
    en, ms = witness_statement(v)
    assert "possibly one of: Perodua Myvi, Perodua Axia, Proton Iriz" in en
    assert "[time not given]" in en and "could not read the registration" in en
    assert "tidak dapat membaca" in ms


def test_statement_without_vehicle():
    en, ms = witness_statement(None, when="today", where="here")
    assert "could not identify" in en and "tidak dapat saya kenal pasti" in ms
