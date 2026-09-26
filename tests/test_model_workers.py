import threading

import pytest

from mlxh.model_workers import ModelWorkers


def test_worker_is_started_once_and_reused_until_idle_expiry():
    started, stopped = [], []
    pool = ModelWorkers(
        lambda name: started.append(name) or object(),
        stopped.append, idle_timeout_s=60, reap_interval_s=60,
    )
    try:
        with pool.lease("alpha") as first:
            with pool.lease("alpha") as second:
                assert second is first
                assert pool.snapshot()[0]["active_requests"] == 2
            assert pool.snapshot()[0]["state"] == "busy"
        assert started == ["alpha"]
        assert pool.snapshot()[0]["state"] == "idle"
        assert pool.reap(now=float("inf")) == ["alpha"]
        assert stopped == [first]
    finally:
        pool.close()


def test_different_models_can_start_concurrently_without_an_implicit_cap():
    barrier = threading.Barrier(3)
    started = []
    lock = threading.Lock()

    def start(name):
        with lock:
            started.append(name)
        barrier.wait(timeout=2)
        return name

    pool = ModelWorkers(start, lambda _worker: None, reap_interval_s=60)
    failures = []

    def acquire(name):
        try:
            with pool.lease(name):
                pass
        except Exception as exc:  # make thread failures visible to pytest
            failures.append(exc)

    threads = [threading.Thread(target=acquire, args=(name,))
               for name in ("alpha", "beta")]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=2)
        assert not failures
        assert set(started) == {"alpha", "beta"}
        assert {worker["model"] for worker in pool.snapshot()} == {"alpha", "beta"}
    finally:
        pool.close()


def test_idle_timeout_never_reaps_a_worker_with_an_active_lease():
    stopped = []
    pool = ModelWorkers(lambda name: name, stopped.append,
                        idle_timeout_s=0, reap_interval_s=60)
    try:
        with pool.lease("alpha"):
            assert pool.reap(now=float("inf")) == []
            assert stopped == []
        assert pool.reap(now=float("inf")) == ["alpha"]
        assert stopped == ["alpha"]
    finally:
        pool.close()


def test_start_failure_leaves_no_loaded_worker():
    pool = ModelWorkers(
        lambda _name: (_ for _ in ()).throw(RuntimeError("load failed")),
        lambda _worker: None, reap_interval_s=60,
    )
    try:
        with pytest.raises(RuntimeError, match="load failed"):
            with pool.lease("alpha"):
                pass
        assert pool.snapshot() == []
    finally:
        pool.close()


def test_exited_worker_is_restarted_on_next_lease():
    stopped = []

    class Process:
        code = None

        def poll(self):
            return self.code

    class Handle:
        def __init__(self):
            self.process = Process()

    started = []

    def start(_name):
        handle = Handle()
        started.append(handle)
        return handle

    pool = ModelWorkers(start, stopped.append, idle_timeout_s=60, reap_interval_s=60)
    try:
        with pool.lease("alpha") as first:
            pass
        first.process.code = 9
        with pool.lease("alpha") as second:
            assert second is not first
        assert len(started) == 2
    finally:
        pool.close()


def test_close_stops_resident_workers_and_rejects_new_leases():
    stopped = []
    pool = ModelWorkers(lambda name: name, stopped.append, reap_interval_s=60)
    with pool.lease("alpha"):
        pass
    pool.close()
    pool.close()
    assert stopped == ["alpha"]
    with pytest.raises(RuntimeError, match="shutting down"):
        with pool.lease("beta"):
            pass
