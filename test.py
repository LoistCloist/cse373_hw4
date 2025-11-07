from pathlib import Path
import subprocess

test_dir = Path(r"C:\Users\heinz\Desktop\CS_Projects\cse373\test_data")
TIMEOUT_SECONDS = 10
for file in sorted(test_dir.glob("*")):
        print(f"\nProcessing: {file.name}")
        try:
                print("this runs")
                result = subprocess.run(
                ["python", "set_cover_reference.py", str(file), str(10)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS
                )
                print(result.stdout)
        except subprocess.TimeoutExpired:
                print("timed out")

# The code snippet you provided is using the `subprocess.run()` function in Python to execute a Python
# script called "set_cover_reference.py" with some arguments. Here's a breakdown of what the code is
# doing:
        # result2 = subprocess.run(
        #     ["python", "set_cover_reference.py", str(file), str(100)],
        #     capture_output=True,
        #     text=True
        # )
        # print(result2.stdout)
# test_file = Path(r"C:\Users\heinz\Desktop\CS_Projects\cse373\test_data\s-rg-31-15")

# print(f"Processing: {test_file.name}")
# result = subprocess.run(
#     ["python", "another.py", str(test_file)],
#     capture_output=True,
#     text=True
# )
# print(result.stdout)
# result2 = subprocess.run(
#     ["python", "set_cover_reference.py", str(test_file), str(100)],
#     capture_output=True,
#     text=True
# )
# print(result2.stdout)
