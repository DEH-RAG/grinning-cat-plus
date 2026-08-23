"""Import-safe, self-contained check for the VLLM embedder's persist behavior.

Verifies ``CustomVllmMultimodalEmbedder._persist_auto_budget_to_config``: the
one-shot, fire-and-forget persistence of an auto-detected ``max_input_tokens``
into the system embedder config via the sync Redis JSON store.

Runs standalone with plain ``python3`` (NOT a pytest-collected module). The
ENTIRE body -- all imports (including the module under test, sys, unittest.mock,
redis, and the fake-redis classes), all fakes, all helpers, and all assert runs
-- lives inside ``if __name__ == "__main__":``. Importing this file therefore
executes ZERO code and emits ZERO stdout, which is the required import-safety
contract for files shipped under a cat plugin folder (the host import_module()s
every .py recursively at activation).
"""

if __name__ == "__main__":
    import redis
    from unittest import mock

    import embedders.custom as embedder_module
    from embedders.custom import CustomVllmMultimodalEmbedder

    KEY = "system:agent"
    PATH = '$[?(@.name=="VllmMultimodalConfiguration")].value.max_input_tokens'
    DETECTED = 12345

    class _JsonHandle:
        """Records redis .json() get/set calls; can simulate a pre-existing
        value (get_result) or a hard Redis failure on set."""

        def __init__(self, get_result=None, fail_set=False):
            self.get_result = get_result
            self.fail_set = fail_set
            self.get_calls = []
            self.set_calls = []

        def get(self, key, path):
            self.get_calls.append((key, path))
            return self.get_result

        def set(self, key, path, value):
            if self.fail_set:
                raise redis.exceptions.RedisError("boom")
            self.set_calls.append((key, path, value))

    class _FakeRedis:
        """Minimal stand-in for the redis client returned by get_sync_db()."""

        def __init__(self, handle):
            self._handle = handle

        def json(self):
            return self._handle

    def _construct_auto() -> CustomVllmMultimodalEmbedder:
        """Build an auto-detect embedder with the raw window stubbed to 12345.

        Stubs ``_fetch_max_model_len`` at class scope so __init__ resolves the
        budget and records ``_last_requested_max_model_len = 12345`` (the same
        path the Todo-3 probe drives), exactly like a real reachable vLLM."""
        with mock.patch.object(
            CustomVllmMultimodalEmbedder,
            "_fetch_max_model_len",
            lambda self: DETECTED,
        ):
            return CustomVllmMultimodalEmbedder(
                base_url="http://127.0.0.1:1",
                model="m",
                max_input_tokens=None,
            )

    def _run_with(get_result=None, fail_set=False, override=None):
        """Run the construction+ensure under a fresh fake get_sync_db()."""
        handle = _JsonHandle(get_result=get_result, fail_set=fail_set)
        fake = _FakeRedis(handle)
        with mock.patch.object(
            embedder_module, "get_sync_db", lambda: fake
        ):
            if override is None:
                e = _construct_auto()
            else:
                e = CustomVllmMultimodalEmbedder(
                    base_url="http://127.0.0.1:1",
                    model="m",
                    max_input_tokens=override,
                )
            e._ensure_max_input_tokens()
        return e, handle

    # (a) AUTO-DETECT path: no override, /v1/models reports 12345, nothing in
    #     redis -> exactly one set is recorded with the exact key / JSONPath
    #     / value, and _persisted flips True.
    e, handle = _run_with(get_result=None, fail_set=False)
    assert handle.set_calls == [
        (KEY, PATH, DETECTED)
    ], f"FAIL auto-write: unexpected set calls {handle.set_calls!r}"
    assert handle.get_calls == [
        (KEY, PATH)
    ], f"FAIL auto-write: expected one get to read guard, got {handle.get_calls!r}"
    assert e._persisted is True, "FAIL auto-write: _persisted should be True"
    print("PASS auto-write")

    # (b) OVERRIDE path: explicit admin override -> persistence is a no-op.
    e, handle = _run_with(get_result=None, override=5000)
    assert handle.set_calls == [], (
        f"FAIL override-no-write: expected zero sets, got {handle.set_calls!r}"
    )
    assert e._persisted is False, (
        "FAIL override-no-write: _persisted should stay False on an override"
    )
    print("PASS override-no-write")

    # (c) REDIS-FAIL path: set() raises -> no exception propagates, embedder
    #     stays un-persisted, and the failure is swallowed (fire-and-forget).
    try:
        e, handle = _run_with(get_result=None, fail_set=True, override=None)
        redis_exception_propagated = False
    except redis.exceptions.RedisError:
        redis_exception_propagated = True
    assert redis_exception_propagated is False, (
        "FAIL redis-failure-safe: RedisError escaped _ensure_max_input_tokens"
    )
    assert e._persisted is False, (
        "FAIL redis-failure-safe: _persisted must stay False on a Redis failure"
    )
    print("PASS redis-failure-safe")

    # (d) GUARD path: redis already holds a value for the JSONPath -> the
    #     auto-detect must NOT clobber it (zero sets) and flags itself done.
    e, handle = _run_with(get_result=50000, override=None)
    assert handle.set_calls == [], (
        f"FAIL guard-read-no-clobber: expected zero sets, got {handle.set_calls!r}"
    )
    assert e._persisted is True, (
        "FAIL guard-read-no-clobber: _persisted should be True when a value "
        "already exists"
    )
    print("PASS guard-read-no-clobber")