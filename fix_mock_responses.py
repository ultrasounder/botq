# save as fix_mock_responses.py
import re

with open('botq_working.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the MockInstrument query method
old_query = '''    def query(self, command: str) -> str:
        """Simulate instrument query response"""
        responses = {
            '*IDN?': f'MOCK,{self.name},123456,1.0.0',
            'MEAS:VOLT?': str(48.0 + np.random.normal(0, 0.1)),
            'MEAS:CURR?': str(5.0 + np.random.normal(0, 0.05)),
            'MEAS:TEMP?': str(25.0 + np.random.normal(0, 1.0)),
            'MEAS:RES?': str(0.05 + np.random.normal(0, 0.001)),
            'SYST:ERR?': '0,"No error"',
        }
        return responses.get(command, '0')'''

new_query = '''    def query(self, command: str) -> str:
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
            return '1.0' '''

content = content.replace(old_query, new_query)

with open('botq_final_working.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Final working version saved as botq_final_working.py")