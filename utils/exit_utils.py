import os
import sys

def atexit_handler(res_dir):
    def write_return_code(return_code, status_file):
        with open(status_file, 'w') as f:
            f.write(str(return_code))

    exc_type, exc_value, exc_traceback = sys.exc_info()
    status_file = os.path.join(res_dir, 'return_code.txt')
    if exc_type is not None:
        write_return_code(1, status_file)
    else:
        write_return_code(0, status_file)