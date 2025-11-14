import AutoDockTools
import os
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--receptor_file", required=True, help="Path to the receptor file")
parser.add_argument("-o", "--output_file", required=True, help="Path to the output file")

args = parser.parse_args()
receptor_file = args.receptor_file
output_file = args.output_file

prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')

cmd = f"python {prepare_receptor} -r {receptor_file} -o {output_file} -A hydrogens"
# cmd = f"python {prepare_receptor} -r {receptor_file} -o {output_file} -A hydrogens"
os.popen(cmd).read()