# save as fix_save_results.py
import re

with open('botq_final_working.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the TestExecutionThread run method to always save results
old_run_end = '''            self.progress_update.emit(100, "Test Suite Complete")
            self.test_suite.teardown()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.test_type.replace(' ', '_')}_{self.serial_number}_{timestamp}.csv"
            filepath = os.path.join('test_results', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.test_suite.save_results(filepath)
            
            self.suite_complete.emit()'''

new_run_end = '''            self.progress_update.emit(100, "Test Suite Complete")
            
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
            
            self.suite_complete.emit()'''

# Replace the entire run method's exception handling
content = re.sub(
    r'(self\.progress_update\.emit\(100, "Test Suite Complete"\).*?self\.suite_complete\.emit\(\))',
    new_run_end,
    content,
    flags=re.DOTALL
)

# Also fix individual test methods to not use pytest.fail (which stops execution)
# Replace pytest.fail with logging and continue
content = content.replace('pytest.fail(f"Test failed: {e}")', 
                         'logger.error(f"Test failed: {e}")')

with open('botq_results_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Results-saving version saved as botq_results_fixed.py")