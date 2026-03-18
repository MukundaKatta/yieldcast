"""Tests for Yieldcast."""
from src.core import Yieldcast
def test_init(): assert Yieldcast().get_stats()["ops"] == 0
def test_op(): c = Yieldcast(); c.track(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Yieldcast(); [c.track() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Yieldcast(); c.track(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Yieldcast(); r = c.track(); assert r["service"] == "yieldcast"
