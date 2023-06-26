"""
Tests for function supporting SACC.

"""

from sacc.tracers import (
    BinZTracer,
    BinLogMTracer,
    BinRichnessTracer,
    SurveyTracer,
)


def test_make_binztracer():
    tracer = BinZTracer.make("bin_z", name="fred", lower=0.5, upper=1.0)
    assert isinstance(tracer, BinZTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "fred"
    assert tracer.lower == 0.5
    assert tracer.upper == 1.0


def test_binztracer_equality():
    a = BinZTracer.make("bin_z", name="fred", lower=0.5, upper=1.0)
    b = BinZTracer.make("bin_z", name="fred", lower=0.5, upper=1.0)
    c = BinZTracer.make("bin_z", name="wilma", lower=0.5, upper=1.0)
    d = BinZTracer.make("bin_z", name="fred", lower=0.6, upper=1.0)
    e = BinZTracer.make("bin_z", name="fred", lower=0.5, upper=1.1)
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e


def test_binztracer_tables():
    a = BinZTracer.make("bin_z", name="fred", lower=0.5, upper=1.0)
    b = BinZTracer.make("bin_z", name="wilma", lower=1.0, upper=1.5)
    tables = BinZTracer.to_tables([a, b])
    assert len(tables) == 1  # all BinZTracers are written to a single table

    d = BinZTracer.from_tables(tables)
    assert len(d) == 2  # this list of tables recovers both BinZTracers
    assert d["fred"] == a
    assert d["wilma"] == b


def test_make_binlogmtracer():
    tracer = BinLogMTracer.make("bin_logm", name="fred", lower=13.0, upper=15.0)
    assert isinstance(tracer, BinLogMTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "fred"
    assert tracer.lower == 13.0
    assert tracer.upper == 15.0


def test_binlogmtracer_equality():
    a = BinLogMTracer.make("bin_logm", name="fred", lower=13.0, upper=15.0)
    b = BinLogMTracer.make("bin_logm", name="fred", lower=13.0, upper=15.0)
    c = BinLogMTracer.make("bin_logm", name="wilma", lower=13.0, upper=15.0)
    d = BinLogMTracer.make("bin_logm", name="fred", lower=14.0, upper=15.0)
    e = BinLogMTracer.make("bin_logm", name="fred", lower=13.0, upper=15.1)
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e


def test_binlogmtracer_tables():
    a = BinLogMTracer.make("bin_logm", name="fred", lower=13.0, upper=15.0)
    b = BinLogMTracer.make("bin_logm", name="wilma", lower=14.0, upper=15.5)
    tables = BinLogMTracer.to_tables([a, b])
    assert len(tables) == 1

    d = BinLogMTracer.from_tables(tables)
    assert len(d) == 2
    assert d["fred"] == a
    assert d["wilma"] == b


def test_make_binrichness_tracer():
    tracer = BinRichnessTracer.make(
        "bin_richness",
        name="barney",
        lower=0.25,
        upper=1.0,
    )
    assert isinstance(tracer, BinRichnessTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "barney"
    assert tracer.upper == 1.0
    assert tracer.lower == 0.25


def test_binrichnesstracer_equality():
    a = BinRichnessTracer.make("bin_richness", name="fred", lower=0.5, upper=1.0)
    b = BinRichnessTracer.make("bin_richness", name="fred", lower=0.5, upper=1.0)
    c = BinRichnessTracer.make("bin_richness", name="wilma", lower=0.5, upper=1.0)
    d = BinRichnessTracer.make("bin_richness", name="fred", lower=0.6, upper=1.0)
    e = BinRichnessTracer.make("bin_richness", name="fred", lower=0.5, upper=1.1)
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e


def test_binrichnesstracer_tables():
    a = BinRichnessTracer.make("bin_richness", name="barney", lower=0.0, upper=0.5)
    b = BinRichnessTracer.make("bin_richness", name="betty", lower=1.25, upper=2.0)
    tables = BinRichnessTracer.to_tables([a, b])
    assert len(tables) == 1
    d = BinRichnessTracer.from_tables(tables)
    assert len(d) == 2  # this list of tables recovers both BinRichnessTracers
    assert d["barney"] == a
    assert d["betty"] == b


def test_make_surveytracer():
    tracer = SurveyTracer.make("survey", name="bullwinkle", sky_area=1.0)
    assert isinstance(tracer, SurveyTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "bullwinkle"
    assert tracer.sky_area == 1.0


def test_surveytracer_equality():
    a = SurveyTracer.make("survey", name="bullwinkle", sky_area=1.0)
    b = SurveyTracer.make("survey", name="bullwinkle", sky_area=1.0)
    c = SurveyTracer.make("survey", name="rocky", sky_area=1.0)
    d = SurveyTracer.make("survey", name="boris", sky_area=2.0)

    assert a == b
    assert a != "bullwinkle"
    assert a != c
    assert a != d


def test_surveytracer_tables():
    a = SurveyTracer.make("survey", name="bullwinkle", sky_area=1.0)
    b = SurveyTracer.make("survey", name="rocky", sky_area=2.0)
    tables = SurveyTracer.to_tables([a, b])
    assert len(tables) == 1
    d = SurveyTracer.from_tables(tables)
    assert len(d) == 2
    assert d["bullwinkle"] == a
    assert d["rocky"] == b
