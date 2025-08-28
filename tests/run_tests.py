#!/usr/bin/env python3
"""
Comprehensive test runner for SwarmSort package.

This script provides different test running modes for development and CI/CD.
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, description, ignore_errors=False):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=not ignore_errors)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.1f}s")
        else:
            print(f"❌ {description} failed with exit code {result.returncode} in {duration:.1f}s")
            if not ignore_errors:
                return False
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed with exception in {duration:.1f}s: {e}")
        if not ignore_errors:
            return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return False
    
    return True


def run_unit_tests(verbose=False, coverage=True):
    """Run unit tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_basic.py", "tests/test_core.py"]
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.append("--tb=short")
    
    cmd.extend(["-m", "unit or not slow"])
    
    if coverage:
        cmd.extend(["--cov=swarmsort", "--cov-report=term-missing"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False, coverage=True):
    """Run integration tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_integration.py"]
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.append("--tb=short")
    
    cmd.extend(["-m", "integration and not slow"])
    
    if coverage:
        cmd.extend(["--cov=swarmsort", "--cov-append", "--cov-report=term-missing"])
    
    return run_command(cmd, "Integration Tests")


def run_performance_tests(benchmark_only=False):
    """Run performance tests and benchmarks."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_performance.py"]
    cmd.extend(["-v", "--tb=short", "-m", "performance"])
    
    if benchmark_only:
        cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])
    
    return run_command(cmd, "Performance Tests")


def run_stress_tests():
    """Run stress and edge case tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_stress_edge_cases.py"]
    cmd.extend(["-v", "--tb=short", "-m", "stress", "--timeout=300"])
    
    return run_command(cmd, "Stress Tests")


def run_embedding_tests(verbose=False):
    """Run embedding-specific tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_embedding_specific.py"]
    
    if verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.append("--tb=short")
    
    cmd.extend(["-m", "embedding"])
    
    return run_command(cmd, "Embedding Tests")


def run_slow_tests():
    """Run slow tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-m", "slow", "--timeout=600"]
    
    return run_command(cmd, "Slow Tests")


def run_linting():
    """Run code linting and formatting checks."""
    commands = [
        ([sys.executable, "-m", "black", "--check", "swarmsort/", "tests/", "examples/"], "Black formatting check"),
        ([sys.executable, "-m", "isort", "--check-only", "swarmsort/", "tests/", "examples/"], "Import sorting check"),
        ([sys.executable, "-m", "flake8", "swarmsort/", "tests/", "examples/"], "Flake8 linting"),
        ([sys.executable, "-m", "mypy", "swarmsort/", "--ignore-missing-imports"], "MyPy type checking"),
    ]
    
    all_passed = True
    for cmd, description in commands:
        if not run_command(cmd, description, ignore_errors=True):
            all_passed = False
    
    return all_passed


def run_security_checks():
    """Run security checks."""
    commands = [
        ([sys.executable, "-m", "bandit", "-r", "swarmsort/"], "Bandit security check"),
        ([sys.executable, "-m", "safety", "check"], "Safety dependency check"),
    ]
    
    all_passed = True
    for cmd, description in commands:
        if not run_command(cmd, description, ignore_errors=True):
            all_passed = False
    
    return all_passed


def run_all_tests(quick=False, skip_slow=False, skip_stress=False):
    """Run all test suites."""
    print("🚀 Running SwarmSort Test Suite")
    print(f"⚙️  Quick mode: {quick}, Skip slow: {skip_slow}, Skip stress: {skip_stress}")
    
    all_passed = True
    
    # Core tests
    if not run_unit_tests(verbose=not quick, coverage=True):
        all_passed = False
    
    if not run_integration_tests(verbose=not quick, coverage=True):
        all_passed = False
    
    if not run_embedding_tests(verbose=not quick):
        all_passed = False
    
    # Optional test suites
    if not quick:
        if not run_performance_tests():
            all_passed = False
        
        if not skip_stress:
            if not run_stress_tests():
                all_passed = False
        
        if not skip_slow:
            if not run_slow_tests():
                all_passed = False
    
    return all_passed


def run_ci_tests():
    """Run tests in CI mode."""
    print("🤖 Running CI Test Suite")
    
    all_passed = True
    
    # Linting first
    if not run_linting():
        print("❌ Linting failed, but continuing with tests...")
        all_passed = False
    
    # Core tests
    if not run_unit_tests(verbose=False, coverage=True):
        all_passed = False
    
    if not run_integration_tests(verbose=False, coverage=True):
        all_passed = False
    
    if not run_embedding_tests(verbose=False):
        all_passed = False
    
    # Performance tests (without benchmarking for speed)
    cmd = [sys.executable, "-m", "pytest", "tests/test_performance.py", "-v", "--tb=short", "-m", "performance and not benchmark"]
    if not run_command(cmd, "Performance Tests (CI Mode)"):
        all_passed = False
    
    return all_passed


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="SwarmSort Test Runner")
    
    parser.add_argument("--mode", choices=[
        "unit", "integration", "performance", "stress", "embedding", "slow", "all", "ci", "lint", "security"
    ], default="all", help="Test mode to run")
    
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (less verbose, skip optional tests)")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress tests")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks for performance tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    # Check if pytest is available
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ pytest not found. Install dependencies with: poetry install --with dev")
        sys.exit(1)
    
    # Run selected test mode
    success = True
    
    if args.mode == "unit":
        success = run_unit_tests(verbose=args.verbose)
    elif args.mode == "integration":
        success = run_integration_tests(verbose=args.verbose)
    elif args.mode == "performance":
        success = run_performance_tests(benchmark_only=args.benchmark_only)
    elif args.mode == "stress":
        success = run_stress_tests()
    elif args.mode == "embedding":
        success = run_embedding_tests(verbose=args.verbose)
    elif args.mode == "slow":
        success = run_slow_tests()
    elif args.mode == "all":
        success = run_all_tests(quick=args.quick, skip_slow=args.skip_slow, skip_stress=args.skip_stress)
    elif args.mode == "ci":
        success = run_ci_tests()
    elif args.mode == "lint":
        success = run_linting()
    elif args.mode == "security":
        success = run_security_checks()
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()