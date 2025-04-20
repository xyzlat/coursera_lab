#!/usr/bin/env python3
"""
Script to run all unit tests for the business forecasting application.
"""

import os
import sys
import unittest
import argparse
import time

def discover_and_run_tests(test_dir=None, pattern="test_*.py", verbosity=2):
    """
    Discover and run tests in the specified directory.
    
    Args:
        test_dir: Directory to search for tests (defaults to 'tests' directory)
        pattern: Pattern to match test files
        verbosity: Verbosity level for test output
        
    Returns:
        Tuple of (success, test_results)
    """
    start_time = time.time()
    
    # If no test directory specified, use the 'tests' directory
    if test_dir is None:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.abspath(os.path.join(current_dir, '..', 'tests'))
    
    print(f"Discovering tests in {test_dir} with pattern '{pattern}'")
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    # Run tests
    print("\n" + "="*80)
    print(f"RUNNING ALL TESTS")
    print("="*80 + "\n")
    
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"TEST SUMMARY")
    print("="*80)
    print(f"Ran {result.testsRun} tests in {duration:.2f} seconds")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return success flag and results
    return result.wasSuccessful(), result

def run_specific_test_modules(test_modules, verbosity=2):
    """
    Run specific test modules.
    
    Args:
        test_modules: List of test module names to run
        verbosity: Verbosity level for test output
        
    Returns:
        Tuple of (success, test_results)
    """
    start_time = time.time()
    
    # Create test suite
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    # Add specified test modules to suite
    for module_name in test_modules:
        try:
            # Import the module
            __import__(module_name)
            module = sys.modules[module_name]
            
            # Add tests from the module
            module_tests = loader.loadTestsFromModule(module)
            suite.addTest(module_tests)
            
            print(f"Added tests from module: {module_name}")
        except (ImportError, KeyError) as e:
            print(f"Error importing module {module_name}: {e}")
    
    # Run tests
    print("\n" + "="*80)
    print(f"RUNNING SELECTED TESTS")
    print("="*80 + "\n")
    
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"TEST SUMMARY")
    print("="*80)
    print(f"Ran {result.testsRun} tests in {duration:.2f} seconds")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return success flag and results
    return result.wasSuccessful(), result

def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run all unit tests for the business forecasting application")
    parser.add_argument("--dir", help="Directory to search for tests")
    parser.add_argument("--pattern", default="test_*.py", help="Pattern to match test files")
    parser.add_argument("--verbosity", type=int, default=2, help="Verbosity level (1-3)")
    parser.add_argument("--module", action='append', help="Specific test module to run (can be used multiple times)")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--models", action="store_true", help="Run model tests only")
    parser.add_argument("--logging", action="store_true", help="Run logging tests only")
    parser.add_argument("--ingestion", action="store_true", help="Run data ingestion tests only")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.module:
        # Run specific modules
        success, _ = run_specific_test_modules(args.module, args.verbosity)
    elif any([args.api, args.models, args.logging, args.ingestion]):
        # Run selected test categories
        test_modules = []
        if args.api:
            test_modules.append("tests.test_api")
        if args.models:
            test_modules.append("tests.test_models")
        if args.logging:
            test_modules.append("tests.test_logging")
        if args.ingestion:
            test_modules.append("tests.test_ingestion")
        
        success, _ = run_specific_test_modules(test_modules, args.verbosity)
    else:
        # Run all tests
        success, _ = discover_and_run_tests(args.dir, args.pattern, args.verbosity)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
