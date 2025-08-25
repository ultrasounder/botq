# test_core/torso_eol_tests.py

import time
from datetime import datetime
from typing import List, Tuple
from PyQt5.QtCore import pyqtSignal

from .base_test_suite import BaseTestSuite
from .data_classes import MeasurementResult, MeasurementLimits, TestExecutionStatus

class TorsoEOLTestSuite(BaseTestSuite):
    def run_all_tests(self, progress_signal: pyqtSignal, result_signal: pyqtSignal):
        """
        Runs all tests for the Torso EOL assembly.
        """
        test_methods: List[Tuple[str, str]] = [
            ('test_power_sequencing', 'Power Sequencing'),
            ('test_voltage_regulation', 'Voltage Regulation'),
            ('test_bms_communication', 'BMS Communication'),
            ('test_thermal_performance', 'Thermal Performance'),
            ('test_emergency_stop', 'Emergency Stop'),
        ]
        
        total_tests = len(test_methods)
        
        for i, (method_name, display_name) in enumerate(test_methods):
            progress = int(((i + 1) / total_tests) * 100)
            progress_signal.emit(progress, f"Running: {display_name}")
            
            test_method = getattr(self, method_name)
            
            try:
                # Assuming test methods now return a MeasurementResult
                result = test_method()
            except Exception as e:
                # Handle unexpected errors gracefully
                result = MeasurementResult(
                    test_name=display_name,
                    status=TestExecutionStatus.ERROR,
                    measured_value=0,
                    limits=MeasurementLimits(0, 0, 0, ""),
                    duration=0,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    error_message=str(e)
                )
            
            self.add_result(result)
            result_signal.emit(result)
            
            time.sleep(0.5)  # Add a slight delay for better GUI visibility

# The individual test methods (e.g., test_power_sequencing)
# would now be refactored to return a MeasurementResult object
# instead of adding it to self.results directly.
# ---------------------------------------------
    # Individual Test Implementations
    # ---------------------------------------------

    def test_power_sequencing(self) -> MeasurementResult:
        start_time = time.time()
        test_name = "Power_Sequencing"
        
        try:
            # Your existing test code...
            self.instruments.send_command('power_supply', 'VOLT 48.0')
            self.instruments.send_command('power_supply', 'CURR 10.0')
            self.instruments.send_command('power_supply', 'OUTP ON')
            time.sleep(0.5)
            v_main = float(self.instruments.send_command('dmm', 'MEAS:VOLT:DC?'))
            
            limits = self.test_limits.get(test_name, MeasurementLimits(0, 0, 0, ""))
            passed = limits.check_limits(v_main)
            
            return MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=v_main,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=self.test_limits.get(test_name, MeasurementLimits(0, 0, 0, "")),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )

    def test_voltage_regulation(self) -> MeasurementResult:
        # Refactor this method to return a MeasurementResult object
        # instead of calling self.add_result()
        # ...
        # pass # Placeholder for the refactored code
        """Test 2.2: Voltage regulation under load"""
        start_time = time.time()
        test_name = "Voltage_Regulation"
        try:
            load_levels = [0, 25, 50, 75, 100] # Percentage of rated load
            voltages = []

            for load in load_levels:
                # Set load on electronic load
                current = load * 0.3 # 30 A max
                self.instruments.send_command('electronic_load', f'CURR {current}')
                self.instruments.send_command('electronic_load', 'INP_ON')
                time.sleep(0.5)

                #Measure voltage
                voltage = float(self.instruments.send_command('dmm', 'MEAS:VOLT:DC?'))
                voltages.append(voltage)
            regulation = (max(voltages) - min(voltages)) / max(voltages) * 100

            limits = self.test_limits.get(test_name, MeasurementLimits(0, 0, 0, ""))
            passed = limits.check_limits(regulation)
            #plotting 
            self.plot_regulation(load_levels, voltages)

            return MeasurementResult(
                test_name=test_name, 
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=regulation,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isofformat()
            )
        except Exception as e:
            return MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=self.test_limits.get(test_name, MeasurementLimits(0, 0, 0, "")),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )




    def test_bms_communication(self) -> MeasurementResult:
        # Refactor this method to return a MeasurementResult object
        # ...
        # pass
        """ Test 3.1 BMS CAN communication"""
        start_time = time.time()
        test_name = "BMS_CAN_COMMUNICATION"
        try:
            # In real implementation
            # we woujld use CAN analyzer to simulate message send and receive
            messages_sent = 100
            messages_received = 98
            success_rate = (messages_received / messages_sent) * 100
            limits = MeasurementLimits(95, 100, 100, "%")
            passed = limits.check_limits(success_rate)
            return MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                measured_value=success_rate,
                limits=limits,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoforamt(),
                error_message=str(e)

            )

    def test_thermal_performance(self) -> MeasurementResult:
        # Refactor this method to return a MeasurementResult object
        # ...
        #pass
        """Test4.1 Thermal performance"""
        start_time = time.time()
        test_name="THERMAL_PERFORMANCE"
        try:
            # Apply full load
            self.instruments.send_command('electronics_load', 'CURR 30')
            self.instruments.send_command('electronics_load', 'INP_ON')
            # monitor temperature every 5 minutes
            temps = []
            for i in range(30):
                temp = float(self.instruments.send_command('dmm', 'MEAS:TEMP?'))
                temps.append(temp)
                time.sleep(1)
                max_temp = max(temps)
                limits = MeasurementLimits(0, 85, 25, "C")
                passed = limits.check_limits(max_temp)
                return MeasurementResult(
                    test_name=test_name,
                    status=TestExecutionStatus.PASSED if passed else TestExecutionStatus.FAILED,
                    measured_value=max_temp,
                    limits=limits,
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()

                )
        except Exception as e:
            return MeasurementResult(
                test_name=test_name,
                status=TestExecutionStatus.ERROR,
                measured_value=0,
                limits=MeasurementLimits(0, 0, 0, ""),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            

    def test_emergency_stop(self) -> MeasurementResult:
        # Refactor this method to return a MeasurementResult object
        # ...
        pass
