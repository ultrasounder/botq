# ============================================================================
# Test Configuration and Data Classes
# ============================================================================
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any 
class TestExecutionStatus(Enum):
    """Test execution status"""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"

@dataclass
class MeasurementLimits:
    """Test limits and criteria"""
    min_value: float
    max_value: float
    nominal: float
    units: str
    
    def check_limits(self, value: float) -> bool:
        """Check if value is within limits"""
        return self.min_value <= value <= self.max_value

@dataclass
class MeasurementResult:
    """Individual test result"""
    test_name: str
    status: TestExecutionStatus
    measured_value: float
    limits: MeasurementLimits
    duration: float
    timestamp: str
    error_message: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export"""
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'measured_value': self.measured_value,
            'min_limit': self.limits.min_value,
            'max_limit': self.limits.max_value,
            'nominal': self.limits.nominal,
            'units': self.limits.units,
            'duration': self.duration,
            'timestamp': self.timestamp,
            'error_message': self.error_message
        }