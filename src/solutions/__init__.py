"""Solution ingestion: PDF worked-solutions -> per-problem SolutionRecord.

Mirrors the src/halliday/ layout (schema + classify + pipeline script) but for
solved-solution text rather than topic tags. Solutions never get exported to
public/data/* - they are read only by the private tutor backend, since a
static public export would let anyone scrape the full answer key.
"""
