"""Breaker state machine — fake clock, every transition, epoch semantics."""

from llm_gateway.reliability.circuit_breaker import BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self._t = 1000.0

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def breaker(**overrides) -> CircuitBreaker:
    clock = overrides.pop("clock", FakeClock())
    kw = dict(window=20, failure_rate=0.5, cooldown_s=30.0, half_open_probes=3, clock=clock)
    kw.update(overrides)
    return CircuitBreaker(**kw)


def trip(b: CircuitBreaker, failures: int = 20) -> None:
    for _ in range(failures):
        epoch = b.allow()
        assert epoch is not None
        b.record_failure(epoch)
    assert b.state is BreakerState.OPEN


def test_trips_at_exactly_half_of_full_window():
    b = breaker()
    for _ in range(10):
        b.record_success(b.allow())
    assert b.state is BreakerState.CLOSED
    for _ in range(9):
        b.record_failure(b.allow())
    assert b.state is BreakerState.CLOSED  # 19 samples — window not full
    b.record_failure(b.allow())  # 20th: 10/20 = 0.5 → trip
    assert b.state is BreakerState.OPEN


def test_below_threshold_never_trips():
    b = breaker()
    for _ in range(11):
        b.record_success(b.allow())
    for _ in range(9):
        b.record_failure(b.allow())
    assert b.state is BreakerState.CLOSED  # 9/20 < 0.5


def test_open_refuses_until_cooldown_fully_elapses():
    clock = FakeClock()
    b = breaker(clock=clock)
    trip(b)
    assert b.allow() is None
    clock.advance(29.9)
    assert b.allow() is None  # still OPEN
    clock.advance(0.1)
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow() is not None


def test_half_open_admits_limited_probes():
    clock = FakeClock()
    b = breaker(clock=clock)
    trip(b)
    clock.advance(31)
    assert b.allow() is not None
    assert b.allow() is not None
    assert b.allow() is not None
    assert b.allow() is None  # 4th probe refused


def test_probe_success_closes_and_clears_window():
    clock = FakeClock()
    b = breaker(clock=clock)
    trip(b)
    clock.advance(31)
    epoch = b.allow()
    b.record_success(epoch)
    assert b.state is BreakerState.CLOSED
    b.record_failure(b.allow())  # window cleared: one failure can't trip
    assert b.state is BreakerState.CLOSED


def test_probe_failure_reopens_with_fresh_cooldown():
    clock = FakeClock()
    b = breaker(clock=clock)
    trip(b)
    clock.advance(31)
    epoch = b.allow()
    b.record_failure(epoch)
    assert b.state is BreakerState.OPEN
    clock.advance(29.9)
    assert b.allow() is None  # fresh cooldown honored
    clock.advance(0.1)
    assert b.allow() is not None


def test_stale_epoch_outcomes_are_dropped_by_construction():
    clock = FakeClock()
    b = breaker(clock=clock)
    trip(b)  # epoch bumped past 0
    clock.advance(31)
    epoch = b.allow()
    b.record_failure(0)  # stale (pre-trip epoch) — must be dropped
    assert b.state is BreakerState.HALF_OPEN  # not re-tripped by a stale outcome
    assert b.allow() is not None  # probes continue
    b.record_failure(epoch)  # valid probe failure → open
    assert b.state is BreakerState.OPEN


def test_outcome_from_closed_epoch_is_stale_after_trip():
    b = breaker()
    epoch = b.allow()
    trip(b)  # transitions bump the epoch
    b.record_failure(epoch)  # pre-trip admission — dropped
    # breaker stays OPEN; cooldown still the original one
    assert b.state is BreakerState.OPEN
