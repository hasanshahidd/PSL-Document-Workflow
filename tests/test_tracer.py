from obs.tracer import span, trace_tree, current_span


def test_nested_spans_record_parent_child_relationship():
    with span("root", k=1) as root:
        root_id = root.span_id
        trace_id = root.trace_id
        with span("child") as child:
            assert child.parent_span_id == root_id
            assert child.trace_id == trace_id
            assert current_span() is child
        assert current_span() is root
    spans = trace_tree(trace_id)
    names = [s["name"] for s in spans]
    assert "root" in names and "child" in names


def test_exception_in_span_is_recorded_as_error_status():
    import pytest
    with pytest.raises(RuntimeError):
        with span("will_fail") as s:
            trace_id = s.trace_id
            raise RuntimeError("boom")
    spans = trace_tree(trace_id)
    failed = [s for s in spans if s["name"] == "will_fail"]
    assert failed and failed[0]["status"] == "error"
    assert "boom" in (failed[0]["error"] or "")
