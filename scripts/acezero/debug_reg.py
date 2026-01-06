import sys
import traceback

try:
    # Set the sys.argv before importing the script's __main__ logic
    # We need to simulate the command line arguments
    import argparse
    from pathlib import Path
    
    # Re-run the main logic from register_mapping.py but catch all exceptions
    # Since register_mapping.py is mostly in if __name__ == '__main__':, 
    # we might need to read it and exec it or just import the functions.
    
    print("Starting register_mapping.py debug wrapper...")
    with open("register_mapping.py", "r") as f:
        code = f.read()
    
    # Remove the if __name__ == '__main__': guard to run it
    code = code.replace("if __name__ == '__main__':", "if True:")
    
    exec(code, {"__name__": "__main__", "__file__": "register_mapping.py"})
    print("Execution finished successfully")

except Exception as e:
    print("!!! CATCHED EXCEPTION !!!")
    traceback.print_exc()
    sys.exit(1)
