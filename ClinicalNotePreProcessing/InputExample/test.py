import os
import sys
import subprocess
import re
import pandas as pd

# Change to the previous folder to import pre-written modules
sys.path.append("../")
# os.chdir("../")
from extract_dates_script import findDates