#!/usr/bin/env python3
"""
SwarmSort Test Runner

This script runs all organized test suites for the SwarmSort package.
Tests are organized by module and functionality for better maintainability.
"""

import sys
import os
import time
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test_suite(test_file, verbose=False):
    """Run a specific test suite and return results."""
    args = [test_file]
    if verbose:
        args.append('-v')
    args.extend(['-x', '--tb=short'])  # Stop on first failure, short traceback

    print(f"\n{'='*60}")
    print(f"Running: {Path(test_file).name}")
    print('='*60)

    start_time = time.time()
    result = pytest.main(args)
    elapsed = time.time() - start_time

    status = "PASSED" if result == 0 else "FAILED"
    print(f"Status: {status} (Time: {elapsed:.2f}s)")

    return result == 0, elapsed


def main():
    """Run all test suites in organized order."""

    print("\n" + "="*60)
    print("SWARMSORT TEST SUITE RUNNER")
    print("="*60)

    # Define test suites in order of importance/dependency
    test_suites = [
        # Unit tests for individual modules
        ("test_kalman_filters.py", "Kalman Filter Tests"),
        ("test_track_state.py", "Track State Management Tests"),
        ("test_cost_computation.py", "Cost Computation Tests"),
        ("test_assignment.py", "Assignment Algorithm Tests"),
        ("test_embedding_history.py", "Embedding History Tests"),

        # Integration tests
        ("test_integration.py", "End-to-End Integration Tests"),
    ]

    # Check if verbose mode requested
    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    # Track results
    results = {}
    total_time = 0

    # Run each test suite
    for test_file, description in test_suites:
        test_path = Path(__file__).parent / test_file

        if not test_path.exists():
            print(f"\nWarning: {test_file} not found, skipping...")
            results[test_file] = (False, 0)
            continue

        print(f"\n{description}")
        passed, elapsed = run_test_suite(str(test_path), verbose)
        results[test_file] = (passed, elapsed)
        total_time += elapsed

        # Stop on critical failure if requested
        if not passed and '--stop-on-failure' in sys.argv:
            print("\nStopping due to test failure (--stop-on-failure)")
            break

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed_count = 0
    failed_count = 0

    for test_file, (passed, elapsed) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:8} {test_file:30} ({elapsed:.2f}s)")
        if passed:
            passed_count += 1
        else:
            failed_count += 1

    print("-"*60)
    print(f"Total: {passed_count} passed, {failed_count} failed")
    print(f"Time: {total_time:.2f}s")

    # Return exit code
    return 0 if failed_count == 0 else 1


def run_specific_test(test_name):
    """Run a specific test or test class."""
    print(f"\nRunning specific test: {test_name}")

    # Find test file containing the test
    test_dir = Path(__file__).parent

    for test_file in test_dir.glob("test_*.py"):
        # Check if test name is in file
        with open(test_file, 'r') as f:
            content = f.read()
            if test_name in content:
                print(f"Found in {test_file.name}")
                args = [str(test_file), f"-k {test_name}", "-v"]
                return pytest.main(args)

    print(f"Test '{test_name}' not found")
    return 1


def run_coverage_report():
    """Run tests with coverage reporting."""
    print("\nRunning tests with coverage...")

    try:
        import coverage
    except ImportError:
        print("Coverage not installed. Install with: pip install coverage")
        return 1

    # Run pytest with coverage
    args = [
        '--cov=src/swarmsort',
        '--cov-report=term-missing',
        '--cov-report=html',
        'tests/'
    ]

    return pytest.main(args)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--coverage':
            sys.exit(run_coverage_report())
        elif sys.argv[1] == '--test':
            if len(sys.argv) > 2:
                sys.exit(run_specific_test(sys.argv[2]))
            else:
                print("Usage: run_all_tests.py --test <test_name>")
                sys.exit(1)
        elif sys.argv[1] == '--help':
            print("SwarmSort Test Runner")
            print("\nUsage:")
            print("  run_all_tests.py              # Run all tests")
            print("  run_all_tests.py --verbose    # Run with verbose output")
            print("  run_all_tests.py --coverage   # Run with coverage report")
            print("  run_all_tests.py --test NAME  # Run specific test")
            print("  run_all_tests.py --stop-on-failure  # Stop on first failure")
            sys.exit(0)

    # Run main test suite
    sys.exit(main())