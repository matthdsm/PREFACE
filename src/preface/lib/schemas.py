import pandera.pandas as pa


class SampleSchema(pa.DataFrameSchema):
    """
    Sample schema for samplesheet entries.
    """
    ff = pa.Column(float, checks=[pa.Check.ge(0.0), pa.Check.le(1.0)], nullable=True)
    filepath = pa.Column(str, checks=[pa.Check.str_matches(r'.+\.bed$')])
    id = pa.Column(str)
    sex = pa.Column(str, checks=pa.Check.isin(["M", "F"]))


class SampleDataSchema(pa.DataFrameSchema):
    """
    Sample data model for samplesheet entries with optional fetal fraction.
    """
    chr = pa.Column(str, checks=pa.Check.str_matches(r'^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y|MT)$'))
    start = pa.Column(int, checks=pa.Check.ge(0), coerce=True)
    end = pa.Column(int, checks=pa.Check.ge(0), coerce=True)
    ratio = pa.Column(float, checks=[pa.Check.ge(0.0), pa.Check.le(1.0)], nullable=True)
