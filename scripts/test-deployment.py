#!/usr/bin/env python3
"""
🧪 Deployment Test Suite
Tests all deployed services to ensure they're working correctly
"""

import requests
import time
import json
from datetime import datetime
import sys
from typing import Dict, List, Any

class DeploymentTester:
    """Test suite for deployed services"""

    def __init__(self):
        self.services = {
            "SCADDA Energy": {"url": "http://localhost:8081", "timeout": 30},
            "SCADDA Power": {"url": "http://localhost:8082", "timeout": 30},
            "SquashPlot Chia": {"url": "http://localhost:8083", "timeout": 30},
            "F2 GPU Optimizer": {"url": "http://localhost:8084", "timeout": 30},
            "prime aligned compute Elysia": {"url": "http://localhost:8085", "timeout": 30},
            "Unified Graph": {"url": "http://localhost:8086", "timeout": 30},
            "Benchmark System": {"url": "http://localhost:8087", "timeout": 30},
            "Chia Analysis": {"url": "http://localhost:8088", "timeout": 30},
            "Decentralized Backend": {"url": "http://localhost:3002", "timeout": 30},
            "Contribution Backend": {"url": "http://localhost:3003", "timeout": 30},
            "CAT Credits Backend": {"url": "http://localhost:3004", "timeout": 30},
            "Social Pub/Sub": {"url": "http://localhost:3005", "timeout": 30}
        }
        self.results = {}

    def test_service(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single service"""
        url = config["url"]
        timeout = config["timeout"]

        result = {
            "name": name,
            "url": url,
            "status": "unknown",
            "response_time": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time

            result["status"] = "success" if response.status_code == 200 else "error"
            result["response_time"] = round(response_time, 2)
            result["status_code"] = response.status_code

            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
            except:
                result["response_data"] = response.text[:200]  # First 200 chars

        except requests.exceptions.ConnectionError:
            result["status"] = "connection_error"
            result["error"] = "Connection refused"
        except requests.exceptions.Timeout:
            result["status"] = "timeout"
            result["error"] = f"Timeout after {timeout}s"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run tests on all services"""
        print("🧪 Starting deployment tests...")
        print("=" * 50)

        all_results = {}
        successful = 0
        failed = 0

        for name, config in self.services.items():
            print(f"Testing {name}...")
            result = self.test_service(name, config)
            all_results[name] = result

            if result["status"] == "success":
                print(f"  ✅ {name}: {result['response_time']}s")
                successful += 1
            else:
                print(f"  ❌ {name}: {result['status']} - {result.get('error', 'Unknown error')}")
                failed += 1

        # Summary
        total = len(self.services)
        success_rate = (successful / total) * 100

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_services": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(success_rate, 1),
            "results": all_results
        }

        print("\n" + "=" * 50)
        print(f"📊 Test Results Summary:")
        print(f"   Total Services: {total}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {success_rate}%")

        if success_rate >= 80:
            print("🎉 Deployment test PASSED!")
        else:
            print("⚠️  Deployment test FAILED - Check failed services")

        return summary

    def test_backend_apis(self) -> Dict[str, Any]:
        """Test backend API endpoints specifically"""
        print("\n🔧 Testing Backend APIs...")

        backend_tests = {
            "Decentralized Backend Health": {
                "url": "http://localhost:3002/health",
                "method": "GET"
            },
            "Decentralized Network Status": {
                "url": "http://localhost:3002/api/network/health",
                "method": "GET"
            },
            "Contribution System Stats": {
                "url": "http://localhost:3003/api/contributions/stats",
                "method": "GET"
            },
            "CAT Credits Market Data": {
                "url": "http://localhost:3004/api/market/data",
                "method": "GET"
            },
            "Social System Stats": {
                "url": "http://localhost:3005/api/social/stats",
                "method": "GET"
            }
        }

        results = {}
        for name, config in backend_tests.items():
            print(f"Testing {name}...")
            try:
                if config["method"] == "GET":
                    response = requests.get(config["url"], timeout=10)
                else:
                    response = requests.post(config["url"], timeout=10)

                results[name] = {
                    "status": "success" if response.status_code in [200, 201] else "error",
                    "status_code": response.status_code,
                    "response_time": round(response.elapsed.total_seconds(), 2)
                }
                print(f"  ✅ {name}: {response.status_code}")

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"  ❌ {name}: {str(e)}")

        return results

def main():
    tester = DeploymentTester()

    # Run basic connectivity tests
    basic_results = tester.run_all_tests()

    # Run backend API tests
    backend_results = tester.test_backend_apis()

    # Combine results
    final_report = {
        "basic_tests": basic_results,
        "backend_tests": backend_results,
        "overall_success": basic_results["success_rate"] >= 80
    }

    # Save results to file
    with open("deployment_test_results.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n💾 Test results saved to deployment_test_results.json")

    # Exit with appropriate code
    sys.exit(0 if final_report["overall_success"] else 1)

if __name__ == "__main__":
    main()
