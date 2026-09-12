from llm_gateway.config import Settings


def test_defaults_run_credential_free():
    """Empty env must boot: mock terminates the chain (ADR-0006)."""
    s = Settings(_env_file=None)
    assert s.mock.enabled is True
    assert s.cache.similarity_threshold == 0.95


def test_misconfigured_near_miss_band_fails_loud():
    """The calibration band must sit below the hit threshold."""
    import pytest

    from llm_gateway.config import CacheSettings

    with pytest.raises(ValueError):
        CacheSettings(similarity_threshold=0.85, near_miss_floor=0.85)
