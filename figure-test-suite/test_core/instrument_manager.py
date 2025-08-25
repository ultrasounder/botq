# ============================================================================
# Instrument Control Layer
# ============================================================================
import pyvisa
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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