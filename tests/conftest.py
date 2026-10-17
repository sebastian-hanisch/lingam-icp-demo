"""Gemeinsame Fixtures: Hand-Instanzen (DEMO-PLAYBOOK §1b) und die Experimente, einmal klein (für Struktur- und Verdikt-Tests) und einmal in voller Größe (nur für test_claims)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def small_lingam():
    import lci_evaluation as ev

    return ev.lingam_experiment(reps=3, notears_reps=1, n=500)


@pytest.fixture(scope="session")
def small_gauss():
    import lci_evaluation as ev

    return ev.gauss_experiment(ts=(0.0, 0.5, 1.0), ns=(100, 300), reps=3)


@pytest.fixture(scope="session")
def small_icp():
    import lci_evaluation as ev

    return ev.icp_experiment(reps=3, n_e=500, nes=(100, 300), alphas=(0.01, 0.05))


@pytest.fixture(scope="session")
def small_limits():
    import lci_evaluation as ev

    return ev.limits_experiment(reps=3)


@pytest.fixture(scope="session")
def full_lingam():
    import lci_evaluation as ev

    return ev.lingam_experiment()


@pytest.fixture(scope="session")
def full_gauss():
    import lci_evaluation as ev

    return ev.gauss_experiment()


@pytest.fixture(scope="session")
def full_icp():
    import lci_evaluation as ev

    return ev.icp_experiment()


@pytest.fixture(scope="session")
def full_limits():
    import lci_evaluation as ev

    return ev.limits_experiment()
