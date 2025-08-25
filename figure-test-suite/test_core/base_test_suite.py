from abc import ABC, abstractmethod
import os
import json
import pandas as pd
from typing import List
from datetime import datetime

from .data_classes import MeasurementResult
from .instrument_manager import InstrumentManager

class BaseTestSuite(ABC):
    def __init__(self, instrument_manager: InstrumentManager, config_file: str = 'config.json'):
        self.instruments = instrument_manager
        self.results: List[MeasurementResult] = []
        self.serial_number = ""
        self.operator_id = ""
        self.config = self._load_config(config_file)
        self.test_limits = {}

    def _load_config(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            return json.load(f)

    @abstractmethod
    def run_all_tests(self):
        """Abstract method to be implemented by subclasses."""
        pass

    def setup(self, serial_number: str, operator_id: str, test_type: str):
        self.serial_number = serial_number
        self.operator_id = operator_id
        self.results = []
        self.test_limits = self.config['test_limits'].get(test_type, {})
        # Connect to specific instruments for this test suite
        instrument_addresses = self.config['instruments'].get(test_type.lower().replace(' ', '_'), {})
        for name, address in instrument_addresses.items():
            self.instruments.connect_instrument(name, address)

    def teardown(self):
        self.instruments.close_all()

    def add_result(self, result: MeasurementResult):
        self.results.append(result)

    def save_results(self, filepath: str):
        if not self.results:
            return
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(filepath, index=False)