# NO INTERNAL REFERENCE
import subprocess
def vmsp(input_file, output_file, min_supp):
    subprocess.call(["java", "-jar", "spmf.jar", "run", "VMSP", input_file, output_file, str(min_supp),str(10),str(1)])
    lines = []
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
    except:
        print("read_output error")

    # decode
    patterns = []
    for line in lines:
        line = line.strip()
        patterns.append(line.split(" -1 "))
    return patterns

if __name__ == "__main__":
    vmsp('example_input.txt','test_20201209.txt',0.5)