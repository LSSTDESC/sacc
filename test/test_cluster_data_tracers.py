"""
Tests for function supporting SACC.

"""

from sacc.tracers import (
    BinZTracer,
    BinLogMTracer,
    BinRichnessTracer,
    BinRadiusTracer,
    SurveyTracer,
)
from astropy.table import Table


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
    table = BinZTracer.to_table([a, b])
    assert len(table) == 2  # all BinZTracers are written to a single table

    d = BinZTracer.from_table(table)
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
    table = BinLogMTracer.to_table([a, b])
    assert len(table) == 2

    d = BinLogMTracer.from_table(table)
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
    table = BinRichnessTracer.to_table([a, b])
    assert len(table) == 2
    d = BinRichnessTracer.from_table(table)
    assert len(d) == 2  # this list of tables recovers both BinRichnessTracers
    assert d["barney"] == a
    assert d["betty"] == b


def test_make_binradiustracer():
    tracer = BinRadiusTracer.make(
        "bin_radius", name="pebbles", lower=1.0, center=2.0, upper=3.0
    )
    assert isinstance(tracer, BinRadiusTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "pebbles"
    assert tracer.lower == 1.0
    assert tracer.center == 2.0
    assert tracer.upper == 3.0

def test_binradiustracer_equality():
    a = BinRadiusTracer.make(
        "bin_radius", name="fred", lower=0.5, center=0.75, upper=1.0
    )
    b = BinRadiusTracer.make(
        "bin_radius", name="fred", lower=0.5, center=0.75, upper=1.0
    )
    c = BinRadiusTracer.make(
        "bin_radius", name="wilma", lower=0.5, center=0.75, upper=1.0
    )
    d = BinRadiusTracer.make(
        "bin_radius", name="fred", lower=0.6, center=0.75, upper=1.0
    )
    e = BinRadiusTracer.make(
        "bin_radius", name="fred", lower=0.5, center=0.8, upper=1.0
    )
    f = BinRadiusTracer.make(
        "bin_radius", name="fred", lower=0.5, center=0.75, upper=1.1
    )
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e
    assert a != f


def test_binradiustracer_tables():
    a = BinRadiusTracer.make(
        "bin_radius", name="pebbles", lower=1.0, center=2.0, upper=3.0
    )
    b = BinRadiusTracer.make(
        "bin_radius", name="bambam", lower=3.0, center=4.0, upper=5.0
    )
    table = BinRadiusTracer.to_table([a, b])
    assert len(table) == 2
    d = BinRadiusTracer.from_table(table)
    assert len(d) == 2
    assert d["pebbles"] == a
    assert d["bambam"] == b


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
    table = SurveyTracer.to_table([a, b])
    assert type(table) == Table
    assert len(table) == 2
    d = SurveyTracer.from_table(table)
    assert len(d) == 2
    assert d["bullwinkle"] == a
    assert d["rocky"] == b
