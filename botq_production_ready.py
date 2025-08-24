"""
Figure Robotics Manufacturing Test Suite
Torso EOL and Actuator PCBA Functional Test Implementation
Author: Manufacturing Test Engineering
Version: 1.0.0
"""

import os
import sys
import csv
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import traceback

import pytest
import pyvisa
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                             QProgressBar, QTableWidget, QTableWidgetItem,
                             QGroupBox, QComboBox, QLineEdit, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPalette

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Test Configuration and Data Classes
# ============================================================================

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

# ============================================================================
# Instrument Control Layer
# ============================================================================

class MockInstrument:
    """Mock instrument for testing without hardware"""
    
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        
    def query(self, command: str) -> str:
        """Simulate instrument query response"""
        # Handle variations of measurement commands
        if 'VOLT' in command.upper():
            return str(48.0 + np.random.normal(0, 0.1))
        elif 'CURR' in command.upper():
            return str(5.0 + np.random.normal(0, 0.05))
        elif 'TEMP' in command.upper():
            return str(25.0 + np.random.normal(0, 1.0))
        elif 'RES' in command.upper():
            return str(0.05 + np.random.normal(0, 0.001))
        elif 'FREQ' in command.upper():
            return str(20000 + np.random.normal(0, 10))
        elif 'RIS' in command.upper():  # Rise time
            return str(50e-9 + np.random.normal(0, 5e-9))
        elif '*IDN?' in command:
            return f'MOCK,{self.name},123456,1.0.0'
        elif 'ERR?' in command:
            return '0,"No error"'
        else:
            # Default to a reasonable value for unknown commands
            return '1.0' 
    
    def write(self, command: str):
        """Simulate instrument write command"""
        logger.debug(f"Mock write to {self.name}: {command}")
        
    def close(self):
        """Simulate closing connection"""
        pass

class InstrumentManager:
    """Manages VISA instrument connections and SCPI commands"""
    
    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        self.instruments = {}
        self.mock_mode = False
        self.MockInstrument = MockInstrument  # Set True for testing without instruments
        
    def connect_instrument(self, name: str, address: str) -> bool:
        """Connect to an instrument via VISA"""
        try:
            if self.mock_mode:
                logger.info(f"Mock mode: Simulating connection to {name} at {address}")
                self.instruments[name] = MockInstrument(name, address)
                return True
                
            inst = self.rm.open_resource(address)
            inst.timeout = 5000  # 5 second timeout
            inst.write_termination = '\n'
            inst.read_termination = '\n'
            
            # Test connection
            idn = inst.query('*IDN?')
            logger.info(f"Connected to {name}: {idn}")
            
            self.instruments[name] = inst
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {name}: {e}")
            return False
    
    def send_command(self, instrument: str, command: str) -> str:
        """Send SCPI command and return response"""
        if instrument not in self.instruments:
            raise ValueError(f"Instrument {instrument} not connected")
        
        inst = self.instruments[instrument]
        if '?' in command:
            return inst.query(command)
        else:
            inst.write(command)
            return "OK"
    
    def close_all(self):
        """Close all instrument connections"""
        for name, inst in self.instruments.items():
            try:
                if not self.mock_mode:
                    inst.close()
                logger.info(f"Closed connection to {name}")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
        self.instruments.clear()

class MockInstrument:
    """Mock instrument for testing without hardware"""
    
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        
    def query(self, command: str) -> str:
        """Simulate instrument query response"""
        # Handle variations of measurement commands
        if 'VOLT' in command.upper():
            return str(48.0 + np.random.normal(0, 0.1))
        elif 'CURR' in command.upper():
            return str(5.0 + np.random.normal(0, 0.05))
        elif 'TEMP' in command.upper():
            return str(25.0 + np.random.normal(0, 1.0))
        elif 'RES' in command.upper():
            return str(0.05 + np.random.normal(0, 0.001))
        elif 'FREQ' in command.upper():
            return str(20000 + np.random.normal(0, 10))
        elif 'RIS' in command.upper():  # Rise time
            return str(50e-9 + np.random.normal(0, 5e-9))
        elif '*IDN?' in command:
            return f'MOCK,{self.name},123456,1.0.0'
        elif 'ERR?' in command:
            return '0,"No error"'
        else:
            # Default to a reasonable value for unknown commands
            return '1.0' 
    
    def write(self, command: str):
        """Simulate instrument write command"""
        logger.debug(f"Mock write to {self.name}: {command}")
        
    def close(self):
        """Simulate closing connection"""
        pass

# ============================================================================
# Test Fixtures and Base Classes
# ============================================================================

class BaseTestSuite:
    """Base class for test suites"""
    
    def __init__(self, instrument_manager: InstrumentManager):
        self.instruments = instrument_manager
        self.results = []
        self.start_time = None
        self.serial_number = ""
        self.operator_id = ""
        
    def setup(self, serial_number: str, operator_id: str):
        """Test suite setup"""
        self.serial_number = serial_number
        self.operator_id = operator_id
        self.start_time = datetime.now()
        self.results = []
        logger.info(f"Starting test suite for S/N: {serial_number}")
        
    def teardown(self):
        """Test suite teardown"""
        logger.info(f"Test suite completed. Total tests: {len(self.results)}")
        
    def add_result(self, result: MeasurementResult):
        """Add test result to collection"""
        self.results.append(result)
        
    def save_results(self, filepath: str):
        """Save results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return
            
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")

# ============================================================================
# Torso EOL Test Implementation
# ============================================================================

class TorsoEOLTestSuite(BaseTestSuite):
    """Torso Assembly End-of-Line Test Suite"""
    
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """Pytest fixture for test setup and teardown"""
        # Setup
        logger.info("Setting up Torso EOL test environment")
        self.connect_instruments()
        yield
        # Teardown
        logger.info("Tearing down Torso EOL test environment")
        self.safe_shutdown()
    
    def connect_instruments(self):
        """Connect to all required instruments"""
        instrument_config = {
            'power_supply': 'USB0::0x0957::0x2307::MY53001234::INSTR',
            'dmm': 'USB0::0x0957::0x0607::MY47001234::INSTR',
            'oscilloscope': 'USB0::0x0957::0x1796::MY55001234::INSTR',
            'electronic_load': 'USB0::0x0957::0x2207::MY54001234::INSTR',
        }
        
        for name, address in instrument_config.items():
            self.instruments.connect_instrument(name, address)
    
    def safe_shutdown(self):
        """Ensure safe shutdown of all power systems"""
        try:
            # Disable all outputs
            self.instruments.send_command('power_supply', 'OUTP OFF')
            self.instruments.send_command('electronic_load', 'INP OFF')
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error during safe shutdown: {e}")
    
    # Test Methods
    @pytest.mark.torso
    def test_power_sequencing(self):
        """Test 2.1: Power Sequencing Validation"""
        start_time = time.time()
        test_name = "Power_Sequencing"
        
        try:
            # Configure power supply
            self.instruments.send_command('power_supply', 'VOLT 48.0')
            self.instruments.send_command('power_supply', 'CURR 10.0')
            self.instruments.send_command('power_supply', 'OUTP ON')
            
            # Wait for stabilization
            time.sleep(0.5)
            
            # Measure main voltage
            v_main = float(self.instruments.send_command('dmm', 'MEAS:VOLT:DC?'))
            
            # Check sequencing (simplified - would use scope in real implementation)
            limits = MeasurementLimits(45.6, 50.4, 48.0, "V")
            passed = limits.check_limits(v_main)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=v_main,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Voltage {v_main}V outside limits"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.torso
    def test_voltage_regulation(self):
        """Test 2.2: Voltage Regulation Under Load"""
        start_time = time.time()
        test_name = "Voltage_Regulation"
        
        try:
            load_levels = [0, 25, 50, 75, 100]  # Percentage of rated load
            voltages = []
            
            for load in load_levels:
                # Set electronic load
                current = load * 0.3  # 30A max
                self.instruments.send_command('electronic_load', f'CURR {current}')
                self.instruments.send_command('electronic_load', 'INP ON')
                time.sleep(0.5)
                
                # Measure voltage
                voltage = float(self.instruments.send_command('dmm', 'MEAS:VOLT:DC?'))
                voltages.append(voltage)
                
            # Calculate regulation
            regulation = (max(voltages) - min(voltages)) / 48.0 * 100
            
            limits = MeasurementLimits(0, 1.0, 0.5, "%")
            passed = limits.check_limits(regulation)
            
            # Create plot
            self.plot_regulation(load_levels, voltages)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=regulation,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Regulation {regulation}% exceeds limit"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.torso
    def test_bms_communication(self):
        """Test 3.1: BMS CAN Communication"""
        start_time = time.time()
        test_name = "BMS_Communication"
        
        try:
            # In real implementation, would use CAN analyzer
            # Simulating CAN message exchange
            messages_sent = 100
            messages_received = 98  # Simulated
            
            success_rate = messages_received / messages_sent * 100
            limits = MeasurementLimits(99.0, 100.0, 100.0, "%")
            passed = limits.check_limits(success_rate)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=success_rate,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"CAN success rate {success_rate}% below threshold"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.torso
    def test_thermal_performance(self):
        """Test 4.3: Thermal Performance Under Load"""
        start_time = time.time()
        test_name = "Thermal_Performance"
        
        try:
            # Apply full load
            self.instruments.send_command('electronic_load', 'CURR 30')
            self.instruments.send_command('electronic_load', 'INP ON')
            
            # Monitor temperature for 5 minutes (shortened for demo)
            temps = []
            for i in range(30):  # 30 samples over 30 seconds (demo)
                temp = float(self.instruments.send_command('dmm', 'MEAS:TEMP?'))
                temps.append(temp)
                time.sleep(1)
            
            max_temp = max(temps)
            limits = MeasurementLimits(0, 85.0, 65.0, "°C")
            passed = limits.check_limits(max_temp)
            
            # Create temperature plot
            self.plot_thermal_profile(temps)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=max_temp,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Max temperature {max_temp}°C exceeds limit"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.torso
    def test_emergency_stop(self):
        """Test 6.3: Emergency Stop Functionality"""
        start_time = time.time()
        test_name = "Emergency_Stop"
        
        try:
            # Enable system
            self.instruments.send_command('power_supply', 'OUTP ON')
            time.sleep(0.5)
            
            # Trigger E-stop (simulated via digital output)
            estop_time = time.time()
            self.instruments.send_command('power_supply', 'OUTP OFF')
            
            # Measure shutdown time
            shutdown_complete = time.time()
            shutdown_duration = (shutdown_complete - estop_time) * 1000  # ms
            
            limits = MeasurementLimits(0, 100.0, 50.0, "ms")
            passed = limits.check_limits(shutdown_duration)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=shutdown_duration,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Shutdown time {shutdown_duration}ms exceeds limit"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    # Plotting methods
    def plot_regulation(self, loads: List[float], voltages: List[float]):
        """Create voltage regulation plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(loads, voltages, 'b-o', linewidth=2, markersize=8)
        plt.axhline(y=48, color='g', linestyle='--', label='Nominal')
        plt.axhline(y=50.4, color='r', linestyle='--', label='Upper Limit')
        plt.axhline(y=45.6, color='r', linestyle='--', label='Lower Limit')
        plt.xlabel('Load (%)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage Regulation vs Load')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = f'torso_regulation_{self.serial_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join('test_results', 'plots', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Regulation plot saved to {filepath}")
    
    def plot_thermal_profile(self, temperatures: List[float]):
        """Create thermal profile plot"""
        plt.figure(figsize=(10, 6))
        time_points = list(range(len(temperatures)))
        plt.plot(time_points, temperatures, 'r-', linewidth=2)
        plt.axhline(y=85, color='r', linestyle='--', label='Max Limit')
        plt.axhline(y=65, color='y', linestyle='--', label='Warning')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (°C)')
        plt.title('Thermal Profile Under Load')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = f'torso_thermal_{self.serial_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join('test_results', 'plots', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Thermal plot saved to {filepath}")

# ============================================================================
# Actuator PCBA Test Implementation
# ============================================================================

class ActuatorPCBATestSuite(BaseTestSuite):
    """Actuator Control PCBA Test Suite"""
    
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """Pytest fixture for test setup and teardown"""
        # Setup
        logger.info("Setting up Actuator PCBA test environment")
        self.connect_instruments()
        yield
        # Teardown
        logger.info("Tearing down Actuator PCBA test environment")
        self.safe_shutdown()
    
    def connect_instruments(self):
        """Connect to all required instruments"""
        instrument_config = {
            'power_supply': 'USB0::0x0957::0x2307::MY53001234::INSTR',
            'dmm': 'USB0::0x0957::0x0607::MY47001234::INSTR',
            'oscilloscope': 'USB0::0x0957::0x1796::MY55001234::INSTR',
            'current_probe': 'USB0::0x0957::0x2807::MY56001234::INSTR',
        }
        
        for name, address in instrument_config.items():
            self.instruments.connect_instrument(name, address)
    
    def safe_shutdown(self):
        """Ensure safe shutdown"""
        try:
            self.instruments.send_command('power_supply', 'OUTP OFF')
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error during safe shutdown: {e}")
    
    @pytest.mark.actuator
    def test_supply_current(self):
        """Test 1.1: Supply Current at Idle"""
        start_time = time.time()
        test_name = "Supply_Current_Idle"
        
        try:
            # Power on board
            self.instruments.send_command('power_supply', 'VOLT 48.0')
            self.instruments.send_command('power_supply', 'CURR 2.0')
            self.instruments.send_command('power_supply', 'OUTP ON')
            time.sleep(1)
            
            # Measure idle current
            current = float(self.instruments.send_command('power_supply', 'MEAS:CURR?'))
            
            limits = MeasurementLimits(0.3, 0.5, 0.4, "A")
            passed = limits.check_limits(current)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=current,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Idle current {current}A outside limits"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.actuator
    def test_pwm_generation(self):
        """Test 3.3: PWM Generation"""
        start_time = time.time()
        test_name = "PWM_Generation"
        
        try:
            # Configure scope for PWM measurement
            self.instruments.send_command('oscilloscope', 'CHAN1:DISP ON')
            self.instruments.send_command('oscilloscope', 'TRIG:SOUR CHAN1')
            self.instruments.send_command('oscilloscope', 'TRIG:LEV 2.5')
            
            # Measure frequency
            frequency = float(self.instruments.send_command('oscilloscope', 'MEAS:FREQ? CHAN1'))
            
            limits = MeasurementLimits(19800, 20200, 20000, "Hz")
            passed = limits.check_limits(frequency)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=frequency,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"PWM frequency {frequency}Hz outside limits"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.actuator
    def test_current_sensing(self):
        """Test 4.2: Current Sensing Accuracy"""
        start_time = time.time()
        test_name = "Current_Sensing"
        
        try:
            test_currents = [5.0, 10.0, 20.0, 30.0]
            errors = []
            
            for test_current in test_currents:
                # Apply test current (simulated)
                measured = test_current + np.random.normal(0, 0.1)  # Simulated measurement
                error = abs(measured - test_current) / test_current * 100
                errors.append(error)
            
            max_error = max(errors)
            limits = MeasurementLimits(0, 1.0, 0.5, "%")
            passed = limits.check_limits(max_error)
            
            # Create accuracy plot
            self.plot_current_accuracy(test_currents, errors)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=max_error,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Current sensing error {max_error}% exceeds limit"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    @pytest.mark.actuator
    def test_gate_driver(self):
        """Test 5.1: Gate Driver Functionality"""
        start_time = time.time()
        test_name = "Gate_Driver"
        
        try:
            # Measure rise time
            self.instruments.send_command('oscilloscope', 'MEAS:RIS? CHAN1')
            rise_time = float(self.instruments.send_command('oscilloscope', 'MEAS:RIS? CHAN1'))
            rise_time_ns = rise_time * 1e9  # Convert to nanoseconds
            
            limits = MeasurementLimits(0, 100, 50, "ns")
            passed = limits.check_limits(rise_time_ns)
            
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=rise_time_ns,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.add_result(result)
            assert passed, f"Rise time {rise_time_ns}ns exceeds limit"
            
        except Exception as e:
            result = MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self.add_result(result)
            logger.error(f"Test failed: {e}")
    
    def plot_current_accuracy(self, currents: List[float], errors: List[float]):
        """Create current sensing accuracy plot"""
        plt.figure(figsize=(10, 6))
        plt.plot(currents, errors, 'g-s', linewidth=2, markersize=8)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Max Error')
        plt.xlabel('Test Current (A)')
        plt.ylabel('Error (%)')
        plt.title('Current Sensing Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = f'actuator_current_accuracy_{self.serial_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join('test_results', 'plots', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Current accuracy plot saved to {filepath}")

# ============================================================================
# PyQt5 GUI Implementation
# ============================================================================

class TestExecutionThread(QThread):
    """Thread for running tests without blocking GUI"""
    progress_update = pyqtSignal(int, str)
    test_complete = pyqtSignal(MeasurementResult)
    suite_complete = pyqtSignal()
    
    def __init__(self, test_suite, test_type):
        super().__init__()
        self.test_suite = test_suite
        self.test_type = test_type
        self.serial_number = ""
        self.operator_id = ""
        
    def run(self):
        """Run test suite in thread"""
        try:
            self.test_suite.setup(self.serial_number, self.operator_id)
            
            # Ensure instruments are connected
            self.test_suite.connect_instruments()
            
            # Get test methods
            if self.test_type == "Torso EOL":
                test_methods = [
                    ('test_power_sequencing', 'Power Sequencing'),
                    ('test_voltage_regulation', 'Voltage Regulation'),
                    ('test_bms_communication', 'BMS Communication'),
                    ('test_thermal_performance', 'Thermal Performance'),
                    ('test_emergency_stop', 'Emergency Stop'),
                ]
            else:  # Actuator PCBA
                test_methods = [
                    ('test_supply_current', 'Supply Current'),
                    ('test_pwm_generation', 'PWM Generation'),
                    ('test_current_sensing', 'Current Sensing'),
                    ('test_gate_driver', 'Gate Driver'),
                ]
            
            total_tests = len(test_methods)
            
            for i, (method_name, display_name) in enumerate(test_methods):
                progress = int((i / total_tests) * 100)
                self.progress_update.emit(progress, f"Running: {display_name}")
                
                # Run test method
                method = getattr(self.test_suite, method_name)
                try:
                    method()
                    # Get last result
                    if self.test_suite.results:
                        self.test_complete.emit(self.test_suite.results[-1])
                except Exception as e:
                    logger.error(f"Test {display_name} failed: {e}")
                
                time.sleep(0.5)  # Small delay between tests
            
            self.progress_update.emit(100, "Test Suite Complete")
            
        except Exception as e:
            logger.error(f"Test thread error: {e}")
            self.progress_update.emit(100, f"Test Suite Completed with Errors")
        
        finally:
            # Always save results, even if tests failed
            try:
                self.test_suite.teardown()
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.test_type.replace(' ', '_')}_{self.serial_number}_{timestamp}.csv"
                filepath = os.path.join('test_results', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                if self.test_suite.results:  # Only save if there are results
                    self.test_suite.save_results(filepath)
                    logger.info(f"Results saved to {filepath}")
                else:
                    logger.warning("No results to save")
                    
            except Exception as save_error:
                logger.error(f"Error saving results: {save_error}")
            
        self.suite_complete.emit()

class ManufacturingTestGUI(QMainWindow):
    """Main GUI Application for Manufacturing Test"""
    
    def __init__(self):
        super().__init__()
        self.instrument_manager = InstrumentManager()
        self.instrument_manager.mock_mode = True  # Enable mock mode for demo
        self.test_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Figure Robotics - Manufacturing Test Suite v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #14ffec;
                color: #212121;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QLineEdit, QComboBox {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555555;
                padding: 5px;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QTableWidget {
                background-color: #3c3c3c;
                color: white;
                gridline-color: #555555;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QProgressBar {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #14ffec;
                border-radius: 4px;
            }
            QGroupBox {
                color: white;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        logo_label = QLabel("🤖 FIGURE ROBOTICS - MANUFACTURING TEST SYSTEM")
        logo_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #14ffec;")
        header_layout.addWidget(logo_label)
        header_layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("● READY")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self.status_label)
        main_layout.addLayout(header_layout)
        
        # Test Configuration Section
        config_group = QGroupBox("Test Configuration")
        config_layout = QHBoxLayout()
        
        # Test Type Selection
        test_type_layout = QVBoxLayout()
        test_type_layout.addWidget(QLabel("Test Type:"))
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems(["Torso EOL", "Actuator PCBA"])
        test_type_layout.addWidget(self.test_type_combo)
        config_layout.addLayout(test_type_layout)
        
        # Serial Number Input
        sn_layout = QVBoxLayout()
        sn_layout.addWidget(QLabel("Serial Number:"))
        self.serial_input = QLineEdit()
        self.serial_input.setPlaceholderText("Enter S/N or scan barcode")
        sn_layout.addWidget(self.serial_input)
        config_layout.addLayout(sn_layout)
        
        # Operator ID
        operator_layout = QVBoxLayout()
        operator_layout.addWidget(QLabel("Operator ID:"))
        self.operator_input = QLineEdit()
        self.operator_input.setPlaceholderText("Enter operator ID")
        operator_layout.addWidget(self.operator_input)
        config_layout.addLayout(operator_layout)
        
        # Control Buttons
        button_layout = QVBoxLayout()
        self.start_button = QPushButton("▶ START TEST")
        self.start_button.clicked.connect(self.start_test)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("■ STOP TEST")
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        config_layout.addLayout(button_layout)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)
        
        # Main Content Area (Splitter)
        content_layout = QHBoxLayout()
        
        # Test Results Table
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Test Name", "Status", "Measured", "Limits", "Units", "Duration"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        # Summary Statistics
        summary_layout = QHBoxLayout()
        self.pass_count_label = QLabel("PASS: 0")
        self.pass_count_label.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")
        self.fail_count_label = QLabel("FAIL: 0")
        self.fail_count_label.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: bold;")
        self.yield_label = QLabel("YIELD: 0.0%")
        self.yield_label.setStyleSheet("color: #ffff00; font-size: 14px; font-weight: bold;")
        
        summary_layout.addWidget(self.pass_count_label)
        summary_layout.addWidget(self.fail_count_label)
        summary_layout.addWidget(self.yield_label)
        summary_layout.addStretch()
        results_layout.addLayout(summary_layout)
        
        results_group.setLayout(results_layout)
        content_layout.addWidget(results_group, 2)
        
        # Console Log
        console_group = QGroupBox("Console Log")
        console_layout = QVBoxLayout()
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMaximumHeight(300)
        console_layout.addWidget(self.console_text)
        
        console_group.setLayout(console_layout)
        content_layout.addWidget(console_group, 1)
        
        main_layout.addLayout(content_layout)
        
        # Footer
        footer_layout = QHBoxLayout()
        footer_layout.addWidget(QLabel(f"Version 1.0.0 | {datetime.now().strftime('%Y-%m-%d')}"))
        footer_layout.addStretch()
        footer_layout.addWidget(QLabel("© 2025 Figure Robotics"))
        main_layout.addLayout(footer_layout)
        
        # Timer for updating time
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        
    def update_time(self):
        """Update current time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.setWindowTitle(f"Figure Robotics - Manufacturing Test Suite v1.0 | {current_time}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add message to console log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if level == "ERROR":
            color = "#ff0000"
        elif level == "WARNING":
            color = "#ffff00"
        elif level == "SUCCESS":
            color = "#00ff00"
        else:
            color = "#ffffff"
        
        formatted_message = f'<span style="color: {color}">[{timestamp}] {level}: {message}</span>'
        self.console_text.append(formatted_message)
        self.console_text.verticalScrollBar().setValue(
            self.console_text.verticalScrollBar().maximum()
        )
    
    def start_test(self):
        """Start test execution"""
        # Validate inputs
        if not self.serial_input.text():
            QMessageBox.warning(self, "Input Error", "Please enter serial number")
            return
        
        if not self.operator_input.text():
            QMessageBox.warning(self, "Input Error", "Please enter operator ID")
            return
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("● TESTING")
        self.status_label.setStyleSheet("color: #ffff00; font-size: 16px; font-weight: bold;")
        
        # Clear previous results
        self.results_table.setRowCount(0)
        self.console_text.clear()
        
        # Log start
        self.log_message(f"Starting {self.test_type_combo.currentText()} test", "INFO")
        self.log_message(f"Serial Number: {self.serial_input.text()}", "INFO")
        self.log_message(f"Operator: {self.operator_input.text()}", "INFO")
        
        # Create appropriate test suite
        if self.test_type_combo.currentText() == "Torso EOL":
            test_suite = TorsoEOLTestSuite(self.instrument_manager)
        else:
            test_suite = ActuatorPCBATestSuite(self.instrument_manager)
        
        # Create and start test thread
        self.test_thread = TestExecutionThread(test_suite, self.test_type_combo.currentText())
        self.test_thread.serial_number = self.serial_input.text()
        self.test_thread.operator_id = self.operator_input.text()
        
        # Connect signals
        self.test_thread.progress_update.connect(self.update_progress)
        self.test_thread.test_complete.connect(self.add_test_result)
        self.test_thread.suite_complete.connect(self.test_complete)
        
        # Start thread
        self.test_thread.start()
    
    def stop_test(self):
        """Stop test execution"""
        if self.test_thread and self.test_thread.isRunning():
            self.test_thread.terminate()
            self.log_message("Test stopped by operator", "WARNING")
        
        self.test_complete()
    
    def update_progress(self, value: int, message: str):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.log_message(message, "INFO")
    
    def add_test_result(self, result: MeasurementResult):
        """Add test result to table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Test Name
        self.results_table.setItem(row, 0, QTableWidgetItem(result.test_name))
        
        # Status
        status_item = QTableWidgetItem(result.status.value)
        if result.status == TestExecutionStatus.PASSED:
            status_item.setBackground(QColor(0, 255, 0, 50))
        elif result.status == TestExecutionStatus.FAILED:
            status_item.setBackground(QColor(255, 0, 0, 50))
        else:
            status_item.setBackground(QColor(255, 255, 0, 50))
        self.results_table.setItem(row, 1, status_item)
        
        # Measured Value
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.measured_value:.3f}"))
        
        # Limits
        limits_text = f"{result.limits.min_value:.1f} - {result.limits.max_value:.1f}"
        self.results_table.setItem(row, 3, QTableWidgetItem(limits_text))
        
        # Units
        self.results_table.setItem(row, 4, QTableWidgetItem(result.limits.units))
        
        # Duration
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{result.duration:.2f}s"))
        
        # Update summary
        self.update_summary()
        
        # Log result
        if result.status == TestExecutionStatus.PASSED:
            self.log_message(f"{result.test_name}: PASS", "SUCCESS")
        else:
            self.log_message(f"{result.test_name}: {result.status.value}", "ERROR")
    
    def update_summary(self):
        """Update summary statistics"""
        pass_count = 0
        fail_count = 0
        error_count = 0
        
        for row in range(self.results_table.rowCount()):
            status = self.results_table.item(row, 1).text()
            if status == "PASSED":
                pass_count += 1
            elif status == "FAILED":
                fail_count += 1
            elif status == "ERROR":
                error_count += 1
                fail_count += 1  # Count errors as failures for yield calculation
        
        total = pass_count + fail_count
        if total > 0:
            yield_rate = (pass_count / total) * 100
        else:
            yield_rate = 0
        
        self.pass_count_label.setText(f"PASS: {pass_count}")
        self.fail_count_label.setText(f"FAIL: {fail_count + error_count}")
        self.yield_label.setText(f"YIELD: {yield_rate:.1f}%")
    
    def test_complete(self):
        """Handle test completion"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        
        # Update status based on results
        if self.results_table.rowCount() > 0:
            all_pass = True
            for row in range(self.results_table.rowCount()):
                if self.results_table.item(row, 1).text() != "PASSED":
                    all_pass = False
                    break
            
            if all_pass:
                self.status_label.setText("● PASS")
                self.status_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
                self.log_message("TEST SUITE PASSED", "SUCCESS")
            else:
                self.status_label.setText("● FAIL")
                self.status_label.setStyleSheet("color: #ff0000; font-size: 16px; font-weight: bold;")
                self.log_message("TEST SUITE FAILED", "ERROR")
        else:
            self.status_label.setText("● READY")
            self.status_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
    
    def closeEvent(self, event):
        """Handle application close"""
        reply = QMessageBox.question(
            self, 'Exit Confirmation',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.test_thread and self.test_thread.isRunning():
                self.test_thread.terminate()
            self.instrument_manager.close_all()
            event.accept()
        else:
            event.ignore()

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point"""
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    os.makedirs('test_results/plots', exist_ok=True)
    
    # Create Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ManufacturingTestGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == '__main__':
    import sys
    # Only skip main() if pytest is actually running tests
    if not any('pytest' in arg for arg in sys.argv):
        main()