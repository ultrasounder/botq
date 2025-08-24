# save as final_fix.py
import re

with open('botq_final.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Move MockInstrument outside of InstrumentManager class
# Find the MockInstrument class definition
mock_class = '''class MockInstrument:
    """Mock instrument for testing without hardware"""
    
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        
    def query(self, command: str) -> str:
        """Simulate instrument query response"""
        responses = {
            '*IDN?': f'MOCK,{self.name},123456,1.0.0',
            'MEAS:VOLT?': str(48.0 + np.random.normal(0, 0.1)),
            'MEAS:CURR?': str(5.0 + np.random.normal(0, 0.05)),
            'MEAS:TEMP?': str(25.0 + np.random.normal(0, 1.0)),
            'MEAS:RES?': str(0.05 + np.random.normal(0, 0.001)),
            'SYST:ERR?': '0,"No error"',
        }
        return responses.get(command, '0')
    
    def write(self, command: str):
        """Simulate instrument write command"""
        logger.debug(f"Mock write to {self.name}: {command}")
        
    def close(self):
        """Simulate closing connection"""
        pass'''

# Add import for np at the top if not present
if 'import numpy as np' not in content:
    content = content.replace('import pandas as pd', 'import pandas as pd\nimport numpy as np')

# Place MockInstrument before InstrumentManager
content = content.replace('class InstrumentManager:', f'{mock_class}\n\nclass InstrumentManager:')

# Fix 2: Update connect_instrument method to properly handle mock mode
old_connect = '''    def connect_instrument(self, name: str, address: str) -> bool:
        """Connect to an instrument via VISA"""
        try:
            if self.mock_mode:
                logger.info(f"Mock mode: Simulating connection to {name} at {address}")
                self.instruments[name] = MockInstrument(name, address)
                return True'''

new_connect = '''    def connect_instrument(self, name: str, address: str) -> bool:
        """Connect to an instrument via VISA"""
        try:
            if self.mock_mode:
                logger.info(f"Mock mode: Simulating connection to {name} at {address}")
                self.instruments[name] = MockInstrument(name, address)
                return True'''

content = content.replace(old_connect, new_connect)

# Fix 3: Ensure connect_instruments is called in TestExecutionThread
thread_fix = '''            # Get test methods
            if self.test_type == "Torso EOL":'''

thread_fix_new = '''            # Ensure instruments are connected
            self.test_suite.connect_instruments()
            
            # Get test methods
            if self.test_type == "Torso EOL":'''

content = content.replace(thread_fix, thread_fix_new)

# Save the fixed file
with open('botq_working.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed version saved as botq_working.py")