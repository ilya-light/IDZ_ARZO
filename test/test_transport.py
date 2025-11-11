
import os
import numpy as np
import pytest
from src.main import TransportProblem, readTransportCSV

# Prepare test CSV files (created dynamically for test isolation)
@pytest.fixture(scope="module")
def setup_test_files(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("transport_tests")

    # Example 1
    csv1 = base_dir / "example1.csv"
    csv1.write_text(
        "7,9,18\n"
        "5,8,7,14\n"
        "19,30,50,10\n"
        "70,30,40,60\n"
        "40,8,70,20\n"
    )

    # Example 2
    csv2 = base_dir / "example2.csv"
    csv2.write_text(
        "20,30,25\n"
        "10,25,15,25\n"
        "8,6,10,9\n"
        "9,12,13,7\n"
        "14,9,16,5\n"
    )

    return base_dir


def test_read_transport_csv(setup_test_files):
    file_path = setup_test_files / "example1.csv"
    costs, supplies, demands = readTransportCSV(file_path)

    assert costs.shape == (3, 4)
    assert len(supplies) == 3
    assert len(demands) == 4
    assert np.isclose(supplies.sum(), 34)
    assert np.isclose(demands.sum(), 34)


def test_vogel_initial_solution(setup_test_files):
    file_path = setup_test_files / "example2.csv"
    costs, supplies, demands = readTransportCSV(file_path)

    tp = TransportProblem(costs, supplies, demands)
    tp.vogelInitialSolution()

    alloc = np.where(np.isnan(tp.allocation), 0.0, tp.allocation)
    # sum of allocations per supplier should not exceed supply
    assert np.allclose(alloc.sum(axis=1), supplies, atol=1e-6) or np.all(alloc.sum(axis=1) <= supplies + 1e-6)
    # sum of allocations per consumer should not exceed demand
    assert np.allclose(alloc.sum(axis=0), demands, atol=1e-6) or np.all(alloc.sum(axis=0) <= demands + 1e-6)
    # should have m+n-1 basic variables
    basic_count = np.count_nonzero(~np.isnan(tp.allocation))
    assert basic_count >= tp.m + tp.n - 1


def test_solve_by_potentials(setup_test_files):
    file_path = setup_test_files / "example1.csv"
    costs, supplies, demands = readTransportCSV(file_path)
    tp = TransportProblem(costs, supplies, demands)
    alloc, total_cost, iterations = tp.solveByPotentials(verbose=False)

    # basic sanity checks
    assert isinstance(total_cost, float)
    assert iterations > 0
    assert alloc.shape == costs.shape
    # solution should satisfy all demands and supplies
    row_sums = np.nansum(alloc, axis=1)
    col_sums = np.nansum(alloc, axis=0)
    assert np.allclose(row_sums, tp.supplies, atol=1e-6)
    assert np.allclose(col_sums, tp.demands, atol=1e-6)