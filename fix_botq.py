# save as fix_botq.py
import re

# Open with UTF-8 encoding to handle special characters
with open('botq.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace class names
replacements = [
    (r'\bTestStatus\b', 'TestExecutionStatus'),
    (r'\bTestLimits\b', 'MeasurementLimits'),
    (r'\bTestResult\b', 'MeasurementResult'),
    (r'\bTestThread\b', 'TestExecutionThread'),
    (r'\bTestManufacturingGUI\b', 'ManufacturingTestGUI'),
]

for old, new in replacements:
    content = re.sub(old, new, content)

# Write with UTF-8 encoding
with open('botq_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed version saved as botq_fixed.py")