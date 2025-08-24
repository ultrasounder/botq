# save as fix_error_count.py
import re

with open('botq_results_fixed.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the update_summary method
old_summary = '''    def update_summary(self):
        """Update summary statistics"""
        pass_count = 0
        fail_count = 0
        
        for row in range(self.results_table.rowCount()):
            status = self.results_table.item(row, 1).text()
            if status == "PASSED":
                pass_count += 1
            elif status == "FAILED":
                fail_count += 1
        
        total = pass_count + fail_count
        if total > 0:
            yield_rate = (pass_count / total) * 100
        else:
            yield_rate = 0
        
        self.pass_count_label.setText(f"PASS: {pass_count}")
        self.fail_count_label.setText(f"FAIL: {fail_count}")
        self.yield_label.setText(f"YIELD: {yield_rate:.1f}%")'''

new_summary = '''    def update_summary(self):
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
        self.yield_label.setText(f"YIELD: {yield_rate:.1f}%")'''

content = content.replace(old_summary, new_summary)

# Save the final version
with open('botq_production_ready.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Production-ready version saved as botq_production_ready.py")