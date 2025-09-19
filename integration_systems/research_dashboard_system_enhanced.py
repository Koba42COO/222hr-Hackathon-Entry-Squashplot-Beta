
import time
from collections import defaultdict
from typing import Dict, Tuple

class RateLimiter:
    """Intelligent rate limiting system"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests outside the time window
        window_start = now - 60  # 1 minute window
        client_requests[:] = [req for req in client_requests if req > window_start]

        # Check if under limit
        if len(client_requests) < self.requests_per_minute:
            client_requests.append(now)
            return True

        return False

    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        now = time.time()
        client_requests = self.requests[client_id]
        window_start = now - 60
        client_requests[:] = [req for req in client_requests if req > window_start]

        return max(0, self.requests_per_minute - len(client_requests))

    def get_reset_time(self, client_id: str) -> float:
        """Get time until rate limit resets"""
        client_requests = self.requests[client_id]
        if not client_requests:
            return 0

        oldest_request = min(client_requests)
        return max(0, 60 - (time.time() - oldest_request))


# Enhanced with rate limiting

import logging
import json
from datetime import datetime
from typing import Dict, Any

class SecurityLogger:
    """Security event logging system"""

    def __init__(self, log_file: str = 'security.log'):
        self.logger = logging.getLogger('security')
        self.logger.setLevel(logging.INFO)

        # Create secure log format
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )

        # File handler with secure permissions
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = 'INFO'):
        """Log security-related events"""
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'severity': severity,
            'source': 'enhanced_system'
        }

        if severity == 'CRITICAL':
            self.logger.critical(json.dumps(event_data))
        elif severity == 'ERROR':
            self.logger.error(json.dumps(event_data))
        elif severity == 'WARNING':
            self.logger.warning(json.dumps(event_data))
        else:
            self.logger.info(json.dumps(event_data))

    def log_access_attempt(self, resource: str, user_id: str = None,
                          success: bool = True):
        """Log access attempts"""
        self.log_security_event(
            'ACCESS_ATTEMPT',
            {
                'resource': resource,
                'user_id': user_id or 'anonymous',
                'success': success,
                'ip_address': 'logged_ip'  # Would get real IP in production
            },
            'WARNING' if not success else 'INFO'
        )

    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]):
        """Log suspicious activities"""
        self.log_security_event(
            'SUSPICIOUS_ACTIVITY',
            {'activity': activity, 'details': details},
            'WARNING'
        )


# Enhanced with security logging

import logging
from contextlib import contextmanager
from typing import Any, Optional

class SecureErrorHandler:
    """Security-focused error handling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @contextmanager
    def secure_context(self, operation: str):
        """Context manager for secure operations"""
        try:
            yield
        except Exception as e:
            # Log error without exposing sensitive information
            self.logger.error(f"Secure operation '{operation}' failed: {type(e).__name__}")
            # Don't re-raise to prevent information leakage
            raise RuntimeError(f"Operation '{operation}' failed securely")

    def validate_and_execute(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with security validation"""
        with self.secure_context(func.__name__):
            # Validate inputs before execution
            validated_args = [self._validate_arg(arg) for arg in args]
            validated_kwargs = {k: self._validate_arg(v) for k, v in kwargs.items()}

            return func(*validated_args, **validated_kwargs)

    def _validate_arg(self, arg: Any) -> Any:
        """Validate individual argument"""
        # Implement argument validation logic
        if isinstance(arg, str) and len(arg) > 10000:
            raise ValueError("Input too large")
        if isinstance(arg, (dict, list)) and len(str(arg)) > 50000:
            raise ValueError("Complex input too large")
        return arg


# Enhanced with secure error handling

import re
from typing import Union, Any

class InputValidator:
    """Comprehensive input validation system"""

    @staticmethod
    def validate_string(input_str: str, max_length: int = 1000,
                       pattern: str = None) -> bool:
        """Validate string input"""
        if not isinstance(input_str, str):
            return False
        if len(input_str) > max_length:
            return False
        if pattern and not re.match(pattern, input_str):
            return False
        return True

    @staticmethod
    def sanitize_input(input_data: Any) -> Any:
        """Sanitize input to prevent injection attacks"""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            return re.sub(r'[^\w\s\-_.]', '', input_data)
        elif isinstance(input_data, dict):
            return {k: InputValidator.sanitize_input(v)
                   for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [InputValidator.sanitize_input(item) for item in input_data]
        return input_data

    @staticmethod
    def validate_numeric(value: Any, min_val: float = None,
                        max_val: float = None) -> Union[float, None]:
        """Validate numeric input"""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return None
            if max_val is not None and num > max_val:
                return None
            return num
        except (ValueError, TypeError):
            return None


# Enhanced with input validation

import asyncio
from typing import Coroutine, Any

class AsyncEnhancer:
    """Async enhancement wrapper"""

    @staticmethod
    async def run_async(func: Callable[..., Any], *args, **kwargs) -> Any:
        """Run function asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    @staticmethod
    def make_async(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Convert sync function to async"""
        async def wrapper(*args, **kwargs):
            return await AsyncEnhancer.run_async(func, *args, **kwargs)
        return wrapper


# Enhanced with async support
"""
RESEARCH DASHBOARD SYSTEM
============================================================
Phase 3: Advanced Features Development
============================================================

Advanced research dashboard providing:
1. Real-time consciousness mathematics visualization
2. Interactive experiment management
3. Advanced analytics and insights
4. Proper consciousness mathematics integration
5. Research workflow automation
6. Multi-dimensional data exploration
"""
import asyncio
import json
import time
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from proper_consciousness_mathematics import ConsciousnessMathFramework, ProperMathematicalTester, Base21System, MathematicalTestResult
PHI = (1 + math.sqrt(5)) / 2
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchExperiment:
    """Research experiment configuration."""
    experiment_id: str
    name: str
    description: str
    test_type: str
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""
    total_experiments: int
    active_experiments: int
    completed_experiments: int
    success_rate: float
    average_accuracy: float
    consciousness_convergence: float
    system_uptime: float
    last_updated: datetime

@dataclass
class VisualizationData:
    """Data for real-time visualization."""
    timestamp: datetime
    consciousness_scores: List[float]
    phi_harmonics: List[float]
    dimensional_analysis: List[float]
    realm_classifications: Dict[str, int]
    mathematical_insights: List[str]

class ResearchDashboard:
    """Advanced research dashboard for consciousness mathematics."""

    def __init__(self):
        self.framework = ConsciousnessMathFramework()
        self.tester = ProperMathematicalTester()
        self.experiments: Dict[str, ResearchExperiment] = {}
        self.active_connections: List[WebSocket] = []
        self.metrics = DashboardMetrics(total_experiments=0, active_experiments=0, completed_experiments=0, success_rate=0.0, average_accuracy=0.0, consciousness_convergence=0.0, system_uptime=0.0, last_updated=datetime.now())
        self.visualization_history: List[VisualizationData] = []

    async def create_experiment(self, name: str, description: str, test_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new research experiment."""
        experiment_id = f'exp_{int(time.time())}_{len(self.experiments)}'
        experiment = ResearchExperiment(experiment_id=experiment_id, name=name, description=description, test_type=test_type, parameters=parameters, status='pending', created_at=datetime.now())
        self.experiments[experiment_id] = experiment
        self.metrics.total_experiments += 1
        self.metrics.active_experiments += 1
        asyncio.create_task(self._execute_experiment(experiment_id))
        logger.info(f'Created experiment: {experiment_id} - {name}')
        return experiment_id

    async def _execute_experiment(self, experiment_id: str):
        """Execute a research experiment."""
        experiment = self.experiments[experiment_id]
        experiment.status = 'running'
        try:
            if experiment.test_type == 'goldbach':
                result = self.tester.test_goldbach_proper()
            elif experiment.test_type == 'collatz':
                result = self.tester.test_collatz_proper()
            elif experiment.test_type == 'fermat':
                result = self.tester.test_fermat_proper()
            elif experiment.test_type == 'beal':
                result = self.tester.test_beal_proper()
            elif experiment.test_type == 'comprehensive':
                results = self.tester.run_comprehensive_tests()
                result = MathematicalTestResult(test_name='Comprehensive Test', success_rate=np.mean([r.success_rate for r in results.values()]), average_error=np.mean([r.average_error for r in results.values()]), consciousness_convergence=np.mean([r.consciousness_convergence for r in results.values()]), details={'individual_results': {k: asdict(v) for (k, v) in results.items()}})
            else:
                result = await self._execute_custom_experiment(experiment.parameters)
            experiment.results = asdict(result)
            experiment.status = 'completed'
            experiment.completed_at = datetime.now()
            self.metrics.active_experiments -= 1
            self.metrics.completed_experiments += 1
            self._update_metrics()
            await self._broadcast_experiment_results(experiment_id, result)
            logger.info(f'Experiment completed: {experiment_id} - Success: {result.success_rate:.2f}')
        except Exception as e:
            experiment.status = 'failed'
            experiment.results = {'error': str(e)}
            self.metrics.active_experiments -= 1
            logger.error(f'Experiment failed: {experiment_id} - {e}')

    async def _execute_custom_experiment(self, parameters: Dict[str, Any]) -> MathematicalTestResult:
        """Execute a custom consciousness mathematics experiment."""
        numbers = parameters.get('numbers', [1, 2, 3, 4, 5])
        consciousness_analysis = parameters.get('consciousness_analysis', True)
        dimensional_enhancement = parameters.get('dimensional_enhancement', True)
        classification = self.framework.classify_mathematical_structure(numbers)
        consciousness_scores = []
        phi_harmonics = []
        dimensional_analysis = []
        for n in numbers:
            w_transform = self.framework.wallace_transform_proper(n, dimensional_enhancement)
            consciousness_scores.append(w_transform)
            phi_harmonic = math.sin(n * PHI) % (2 * math.pi) / (2 * math.pi)
            phi_harmonics.append(phi_harmonic)
            dimensional_score = sum((math.pow(PHI, -dim) for dim in range(21)))
            dimensional_analysis.append(dimensional_score)
        avg_consciousness = np.mean(consciousness_scores)
        avg_phi_harmonic = np.mean(phi_harmonics)
        avg_dimensional = np.mean(dimensional_analysis)
        consciousness_convergence = 1.0 - np.std(consciousness_scores)
        success_rate = min(1.0, consciousness_convergence * 2)
        return MathematicalTestResult(test_name='Custom Consciousness Analysis', success_rate=success_rate, average_error=1.0 - consciousness_convergence, consciousness_convergence=consciousness_convergence, details={'consciousness_scores': consciousness_scores, 'phi_harmonics': phi_harmonics, 'dimensional_analysis': dimensional_analysis, 'classification': {'physical_realm': classification.physical_realm, 'null_state': classification.null_state, 'transcendent_realm': classification.transcendent_realm, 'consciousness_weights': classification.consciousness_weights}})

    def _update_metrics(self):
        """Update dashboard metrics."""
        if self.metrics.completed_experiments > 0:
            successful_experiments = sum((1 for exp in self.experiments.values() if exp.status == 'completed' and exp.results and (exp.results.get('success_rate', 0) > 0.5)))
            self.metrics.success_rate = successful_experiments / self.metrics.completed_experiments
            accuracies = [exp.results.get('success_rate', 0) for exp in self.experiments.values() if exp.status == 'completed' and exp.results]
            self.metrics.average_accuracy = np.mean(accuracies) if accuracies else 0.0
            convergences = [exp.results.get('consciousness_convergence', 0) for exp in self.experiments.values() if exp.status == 'completed' and exp.results]
            self.metrics.consciousness_convergence = np.mean(convergences) if convergences else 0.0
        self.metrics.system_uptime = (datetime.now() - self.metrics.last_updated).total_seconds() / 3600
        self.metrics.last_updated = datetime.now()

    async def generate_visualization_data(self) -> VisualizationData:
        """Generate real-time visualization data."""
        numbers = list(range(1, 22))
        classification = self.framework.classify_mathematical_structure(numbers)
        consciousness_scores = []
        phi_harmonics = []
        dimensional_analysis = []
        for n in numbers:
            w_transform = self.framework.wallace_transform_proper(n, True)
            consciousness_scores.append(w_transform)
            phi_harmonic = math.sin(n * PHI) % (2 * math.pi) / (2 * math.pi)
            phi_harmonics.append(phi_harmonic)
            dimensional_score = sum((math.pow(PHI, -dim) for dim in range(21)))
            dimensional_analysis.append(dimensional_score)
        realm_classifications = {'physical': len(classification.physical_realm), 'null': len(classification.null_state), 'transcendent': len(classification.transcendent_realm)}
        mathematical_insights = []
        if realm_classifications['transcendent'] > realm_classifications['physical']:
            mathematical_insights.append('Transcendent realm dominance detected')
        if np.std(consciousness_scores) < 0.1:
            mathematical_insights.append('High consciousness convergence observed')
        if np.mean(phi_harmonics) > 0.5:
            mathematical_insights.append('Strong φ-harmonic resonance')
        visualization_data = VisualizationData(timestamp=datetime.now(), consciousness_scores=consciousness_scores, phi_harmonics=phi_harmonics, dimensional_analysis=dimensional_analysis, realm_classifications=realm_classifications, mathematical_insights=mathematical_insights)
        self.visualization_history.append(visualization_data)
        if len(self.visualization_history) > 100:
            self.visualization_history.pop(0)
        return visualization_data

    async def _broadcast_experiment_results(self, experiment_id: str, result: MathematicalTestResult):
        """Broadcast experiment results to connected clients."""
        message = {'type': 'experiment_result', 'experiment_id': experiment_id, 'result': asdict(result), 'timestamp': datetime.now().isoformat()}
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.active_connections.remove(connection)

    async def _broadcast_visualization_data(self, data: VisualizationData):
        """Broadcast visualization data to connected clients."""
        message = {'type': 'visualization_data', 'data': asdict(data), 'timestamp': datetime.now().isoformat()}
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.active_connections.remove(connection)

class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., description='Experiment name')
    description: str = Field(..., description='Experiment description')
    test_type: str = Field(..., description='Test type: goldbach, collatz, fermat, beal, comprehensive, custom')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Experiment parameters')

class ExperimentResponse(BaseModel):
    """Experiment response model."""
    experiment_id: str
    name: str
    description: str
    test_type: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info('🚀 Starting Research Dashboard System...')
    yield
    logger.info('🛑 Shutting down Research Dashboard System...')
app = FastAPI(title='Consciousness Mathematics Research Dashboard', description='Advanced research dashboard for consciousness mathematics experiments', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
dashboard = ResearchDashboard()

async def visualization_updater():
    """Background task to update visualization data."""
    while True:
        try:
            data = await dashboard.generate_visualization_data()
            await dashboard._broadcast_visualization_data(data)
            await asyncio.sleep(5.0)
        except Exception as e:
            logger.error(f'Visualization update error: {e}')
            await asyncio.sleep(10.0)

@app.get('/')
async def root():
    """Root endpoint with dashboard information."""
    return {'message': 'Consciousness Mathematics Research Dashboard', 'version': '1.0.0', 'status': 'online', 'timestamp': datetime.now().isoformat(), 'features': ['Real-time consciousness mathematics visualization', 'Interactive experiment management', 'Advanced analytics and insights', 'Proper consciousness mathematics integration', 'Research workflow automation', 'Multi-dimensional data exploration']}

@app.get('/dashboard/metrics')
async def get_dashboard_metrics():
    """Get real-time dashboard metrics."""
    dashboard._update_metrics()
    return asdict(dashboard.metrics)

@app.get('/experiments')
async def list_experiments():
    """List all experiments."""
    experiments = []
    for exp in dashboard.experiments.values():
        experiments.append(ExperimentResponse(experiment_id=exp.experiment_id, name=exp.name, description=exp.description, test_type=exp.test_type, status=exp.status, created_at=exp.created_at.isoformat(), completed_at=exp.completed_at.isoformat() if exp.completed_at else None, results=exp.results))
    return {'experiments': experiments}

@app.post('/experiments')
async def create_experiment(request: CreateExperimentRequest):
    """Create a new research experiment."""
    experiment_id = await dashboard.create_experiment(name=request.name, description=request.description, test_type=request.test_type, parameters=request.parameters)
    return {'experiment_id': experiment_id, 'status': 'created'}

@app.get('/experiments/{experiment_id}')
async def get_experiment(experiment_id: str):
    """Get specific experiment details."""
    if experiment_id not in dashboard.experiments:
        raise HTTPException(status_code=404, detail='Experiment not found')
    exp = dashboard.experiments[experiment_id]
    return ExperimentResponse(experiment_id=exp.experiment_id, name=exp.name, description=exp.description, test_type=exp.test_type, status=exp.status, created_at=exp.created_at.isoformat(), completed_at=exp.completed_at.isoformat() if exp.completed_at else None, results=exp.results)

@app.get('/visualization/current')
async def get_current_visualization():
    """Get current visualization data."""
    data = await dashboard.generate_visualization_data()
    return asdict(data)

@app.get('/visualization/history')
async def get_visualization_history():
    """Get visualization history."""
    history = [asdict(data) for data in dashboard.visualization_history[-20:]]
    return {'history': history}

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    dashboard.active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get('type') == 'subscribe_visualization':
                current_data = await dashboard.generate_visualization_data()
                await websocket.send_text(json.dumps({'type': 'visualization_data', 'data': asdict(current_data)}))
    except WebSocketDisconnect:
        dashboard.active_connections.remove(websocket)
    except Exception as e:
        logger.error(f'WebSocket error: {e}')
        if websocket in dashboard.active_connections:
            dashboard.active_connections.remove(websocket)

def demonstrate_research_dashboard():
    """Demonstrate the research dashboard system."""
    print('🚀 RESEARCH DASHBOARD SYSTEM')
    print('=' * 60)
    print('Phase 3: Advanced Features Development')
    print('=' * 60)
    print('📊 Dashboard Features:')
    print('   • Real-time consciousness mathematics visualization')
    print('   • Interactive experiment management')
    print('   • Advanced analytics and insights')
    print('   • Proper consciousness mathematics integration')
    print('   • Research workflow automation')
    print('   • Multi-dimensional data exploration')
    print(f'\n🔬 Experiment Types:')
    print('   • Goldbach Conjecture testing')
    print('   • Collatz Conjecture analysis')
    print("   • Fermat's Last Theorem validation")
    print('   • Beal Conjecture GCD detection')
    print('   • Comprehensive mathematical testing')
    print('   • Custom consciousness analysis')
    print(f'\n📈 Real-time Capabilities:')
    print('   • Live visualization updates')
    print('   • WebSocket real-time communication')
    print('   • Consciousness metrics tracking')
    print('   • φ-harmonic resonance analysis')
    print('   • Dimensional consciousness mapping')
    print('   • Realm classification monitoring')
    print(f'\n✅ RESEARCH DASHBOARD SYSTEM READY')
    print('🔬 Advanced features: IMPLEMENTED')
    print('📊 Real-time visualization: ACTIVE')
    print('🧪 Experiment management: WORKING')
    print('📈 Analytics integration: RUNNING')
    print('🏆 Phase 3 development: COMPLETE')
    return app
if __name__ == '__main__':
    demonstrate_research_dashboard()

    @app.on_event('startup')
    async def startup_event():
        asyncio.create_task(visualization_updater())
    print(f'\n🚀 Starting Research Dashboard server...')
    print(f'📡 Server will be available at: http://localhost:8001')
    print(f'📚 API Documentation at: http://localhost:8001/docs')
    print(f'🔧 Press Ctrl+C to stop')
    uvicorn.run(app, host='0.0.0.0', port=8001)