"""Tests for unpii regex patterns — dates, birthdate, email, spans."""

import unpii
import pytest


def spans(text, mode="standard"):
    return [(text[s.start:s.end], s.category) for s in unpii.find_spans(text, mode=mode)]


def categories(text, mode="standard"):
    return [s.category for s in unpii.find_spans(text, mode=mode)]


# ── EMAIL ────────────────────────────────────────────────────────────────────


class TestEmail:
    def test_simple_email(self):
        assert spans("contact: user@gmail.com ok") == [("user@gmail.com", "EMAIL")]

    def test_email_with_dots(self):
        assert spans("a.b.c@test.fr") == [("a.b.c@test.fr", "EMAIL")]

    def test_single_char_local(self):
        assert spans("B@tutanota.com") == [("B@tutanota.com", "EMAIL")]

    def test_dots_before_email_not_captured(self):
        """Dots before local part should not be included."""
        result = spans("...B@tutanota.com end")
        assert len(result) == 1
        assert result[0][0] == "B@tutanota.com"

    def test_email_paranoid(self):
        assert "EMAIL" in categories("foo@bar", mode="paranoid")


# ── DATE ─────────────────────────────────────────────────────────────────────


class TestDate:
    def test_dd_mm_yyyy(self):
        assert spans("le 12/03/1980 ok") == [("12/03/1980", "DATE")]

    def test_dd_mm_yy(self):
        assert spans("le 12/03/80 ok") == [("12/03/80", "DATE")]

    def test_literal_month(self):
        assert spans("le 8 juillet 2020 fin") == [("8 juillet 2020", "DATE")]

    def test_ordinal_day(self):
        assert spans("le 23e avril 1948 fin") == [("23e avril 1948", "DATE")]

    def test_month_day_comma_year(self):
        assert spans("le septembre 29, 2000 fin") == [("septembre 29, 2000", "DATE")]

    def test_month_ordinal_day_comma_year(self):
        assert spans("le mars 11e, 1965 fin") == [("mars 11e, 1965", "DATE")]

    def test_month_slash_yy(self):
        assert spans("en juin/47 ok") == [("juin/47", "DATE")]

    def test_month_slash_yy_long(self):
        assert spans("en novembre/70 ok") == [("novembre/70", "DATE")]

    def test_iso_date(self):
        assert spans("date 2020-03-20 fin") == [("2020-03-20", "DATE")]

    def test_iso_timestamp(self):
        assert spans("date 2002-07-25T00:00:00 fin") == [("2002-07-25T00:00:00", "DATE")]

    def test_month_year(self):
        assert spans("en février 2020 ok") == [("février 2020", "DATE")]

    def test_day_month_no_year(self):
        assert spans("le 20 mars prochain") == [("20 mars", "DATE")]


# ── BIRTHDATE ────────────────────────────────────────────────────────────────


class TestBirthdate:
    """BIRTHDATE and DATE overlap — both categories are acceptable."""

    def test_nee_le_dd_mm_yyyy(self):
        result = spans("née le 12/03/1980 ok")
        assert result[0][0] == "12/03/1980"
        assert result[0][1] in ("BIRTHDATE", "DATE")

    def test_ne_le_literal_month(self):
        result = spans("né le 3 janvier 2020 fin")
        assert result[0][0] == "3 janvier 2020"

    def test_ne_le_ordinal(self):
        result = spans("né le 23e avril 1948 fin")
        assert result[0][0] == "23e avril 1948"

    def test_ne_le_month_first(self):
        result = spans("née le septembre 29, 2000 fin")
        assert result[0][0] == "septembre 29, 2000"

    def test_ne_le_month_slash_yy(self):
        result = spans("né le juin/47 fin")
        assert result[0][0] == "juin/47"

    def test_ne_le_iso_timestamp(self):
        result = spans("née le 2002-07-25T00:00:00 fin")
        assert result[0][0] == "2002-07-25T00:00:00"

    def test_date_de_naissance_dd_mm_yyyy(self):
        result = spans("Date de naissance: 05/05/1959 fin")
        assert result[0][0] == "05/05/1959"

    def test_date_de_naissance_literal(self):
        result = spans("Date de naissance: mars 11e, 1965 fin")
        assert result[0][0] == "mars 11e, 1965"

    def test_date_de_naissance_month_slash(self):
        result = spans("Date de naissance: décembre/04 fin")
        assert result[0][0] == "décembre/04"

    def test_date_de_naissance_iso(self):
        result = spans("Date de naissance: 2002-07-25T00:00:00 fin")
        assert result[0][0] == "2002-07-25T00:00:00"


# ── CHAR OFFSETS (UTF-8 correctness) ─────────────────────────────────────────


class TestCharOffsets:
    def test_accent_before_email(self):
        text = "né test@x.com fin"
        result = unpii.find_spans(text)
        assert len(result) == 1
        s = result[0]
        assert text[s.start:s.end] == "test@x.com"

    def test_multiple_accents(self):
        text = "éàü le 12/03/2020 fin"
        result = unpii.find_spans(text)
        assert len(result) == 1
        s = result[0]
        assert text[s.start:s.end] == "12/03/2020"

    def test_emoji_offset(self):
        text = "hello 😀 user@test.com fin"
        result = unpii.find_spans(text)
        assert len(result) == 1
        s = result[0]
        assert text[s.start:s.end] == "user@test.com"
