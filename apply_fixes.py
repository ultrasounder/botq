# save as apply_fixes.py
import re

# Read the original file
with open('botq_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Make MockInstrument accessible
content = re.sub(
    r'(class InstrumentManager:.*?\n.*?def __init__\(self\):.*?\n.*?self\.mock_mode = False)',
    r'\1\n        self.MockInstrument = MockInstrument',
    content,
    flags=re.DOTALL
)

# Fix 2: Fix the main entry point
old_main = """if __name__ == '__main__':
    # Check if running pytest or GUI
    if 'pytest' in sys.modules:
        # Running tests via pytest
        pass
    else:
        # Running GUI application
        main()"""

new_main = """if __name__ == '__main__':
    import sys
    # Only skip main() if pytest is actually running tests
    if not any('pytest' in arg for arg in sys.argv):
        main()"""

content = content.replace(old_main, new_main)

# Write the fixed file
with open('botq_final.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed version saved as botq_final.py")