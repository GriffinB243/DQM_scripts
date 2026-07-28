import os
import numpy as np
import time
import re
import pandas as pd

def load_csv(path):
    result = pd.read_csv(path)
    return result

outfolder_r0 = '/data/user/fbivens5020/mock_data/' #THIS SHOULD BE ONLY A FOLDER UNDER YOUR USER
outfolder_npys = outfolder_r0 #THIS SHOULD BE ONLY A FOLDER UNDER YOUR USER
time_per_subrun = 5 # IN SECONDS
run = 400215
end_subrun = 15

os.system(f'rm -r {outfolder_r0}*')
os.system(f'rm -r {outfolder_npys}*')

r0_filename = '/data/wipac/CTA/targetcdata/run{0}_subrun{1}_r0.tio'
log_file = '/data/wipac/CTA/targetcdata/run{0}_log.log'

mod_config = load_csv('./module_config.csv')

fee_temp_file = outfolder_npys + 'temperatures_FEEs_run{0}_subrun{1}.npy'
fpm_temp_file = outfolder_npys + 'temperatures_FPMs_run{0}_subrun{1}.npy'
fpm_hv_file = outfolder_npys + 'HV_FPMs_run{0}_subrun{1}.npy'
fee_current_file = outfolder_npys + 'Current_FEEs_run{0}_subrun{1}.npy'

##### FUNCTIONS TO PARSE PROPER TEMPERATURES AND CURRENTS, HVs
def _wildcard_to_regex_parts(pattern: str):
    """
    Split a wildcard pattern (using '*' as "match anything") into a list
    of escaped literal regex fragments, ready to be joined with '.*?'.

    Runs of whitespace in each literal segment are turned into '\\s+' so
    that inconsistent spacing in the source file doesn't break matching.
    """
    parts = pattern.split('*')
    escaped_parts = []
    for part in parts:
        part = part.strip()
        if part == '':
            continue
        escaped = re.escape(part)
        escaped = re.sub(r'(\\\s)+', r'\\s+', escaped)
        escaped_parts.append(escaped)
    return escaped_parts


def _block_pattern_to_regex(pattern: str) -> "re.Pattern":
    """
    Convert a wildcard block-start pattern into a compiled regex that
    simply matches the line (no value capture involved).
    """
    parts = _wildcard_to_regex_parts(pattern)
    regex_str = r'.*?'.join(parts)
    return re.compile(regex_str)


def _pattern_to_regex(pattern: str) -> "re.Pattern":
    """
    Convert a wildcard pattern into a compiled regex that captures the
    number immediately following the pattern's last literal token.

    '*' means "match anything (non-greedily)".
    """
    parts = _wildcard_to_regex_parts(pattern)
    regex_str = r'.*?'.join(parts) + r'\s+([-+]?\d+(?:\.\d+)?)'
    return re.compile(regex_str)


def extract_values(filepath, block_start_pattern, value_patterns):
    """
    Parameters
    ----------
    filepath : str
        Path to the text file to parse.
    block_start_pattern : str
        Wildcard pattern identifying the line that starts a new block,
        e.g. "TEXT 0 - FITH BLOCK" or "TEXT * - FITH BLOCK" (use '*'
        to match variable text, same convention as `value_patterns`).
    value_patterns : dict[str, str]
        Mapping of value-name -> wildcard pattern describing where to
        find that value on a line, e.g.
            {
                "value1": "* SOMETHING - THING 100 Value1",
                "value2": "* SOMETHING - THING 100 Value1 * Meh, Value2",
            }

    Returns
    -------
    list[dict]
        One dict per block found in the file. Each dict contains:
          - "block_header": the raw line that opened the block
          - one key per entry in `value_patterns`, whose value is a list
            of floats (one per line in the block where that pattern matched)
    """
    block_start_re = _block_pattern_to_regex(block_start_pattern)
    compiled = {name: _pattern_to_regex(p) for name, p in value_patterns.items()}

    blocks = []
    current_block = None
    found = False
    with open(filepath, 'r') as f:
        for raw_line in f:
            if not found:
                line = raw_line.rstrip('\n')

                if block_start_re.search(line):
                    current_block = {'block_header': line}
                    for name in value_patterns:
                        current_block[name] = []
                    blocks.append(current_block)
                    continue

                if current_block is None:
                    continue  # ignore anything before the first block
                found = False
                for name, regex in compiled.items():
                    m = regex.search(line)
                    if m:
                        current_block[name].append(float(m.group(1)))
                        found = True
                        pass
                    
                    if found: pass
                    

    return blocks


def generate_temps_FEEs(run, subrun):
    # temps = np.random.normal(35, 3, (22*2))

    patternsPri = {"Pri_mod{}".format(mod): "* - DEBUG    - Module {0} Pri Temp: ".format(mod) for mod in mod_config['module_id']}
    blocksPri = extract_values(
                filepath='/data/wipac/CTA/targetcdata/run{0}_log.log'.format(run),
                block_start_pattern="* - main            - INFO     - Run {0} Sub-Run {1} has * events, total so far in run is *".format(run, int(subrun)-1) if int(subrun) > 0 else "* - main            - INFO     - Run Number: {0}".format(run),  # '*' matches the block number
                value_patterns=patternsPri,
            )

    tempsPri = [blocksPri[0][key][0] for ind, key in enumerate(blocksPri[0].keys()) if ind != 0 and len(blocksPri[0][key]) > 0]

    patternsAux = {"Pri_mod{}".format(mod): "* - DEBUG    - Module {0} Pri Temp: * C, Aux Temp ".format(mod) for mod in mod_config['module_id']}
    blocksAux = extract_values(
                filepath='/data/wipac/CTA/targetcdata/run{0}_log.log'.format(run),
                block_start_pattern="* - main            - INFO     - Run {0} Sub-Run {1} has * events, total so far in run is *".format(run, int(subrun)-1) if int(subrun) > 0 else "* - main            - INFO     - Run Number: {0}".format(run),  # '*' matches the block number
                value_patterns=patternsAux,
            )

    tempsAux = [blocksAux[0][key][0] for ind, key in enumerate(blocksAux[0].keys()) if ind != 0 and len(blocksAux[0][key]) > 0]
    temps = [[tempsPri, tempsAux][l%2][l//2] for l in range(2*len(tempsPri))]

    # temps = []
    try:
        np.save(fee_temp_file.format(run, subrun), temps)
    except:
        pass

def generate_temps_FPMs(run, subrun):
    temps = np.random.normal(30, 3, (22*4))
    try:
        np.save(fpm_temp_file.format(run, subrun), temps)
    except:
        pass

def generate_HV_current(run, subrun):
    # hv = np.random.normal(33, 1, (22*1))
    # current = np.random.normal(0.1, 0.02, (22*1))

    patternsHV = {"Pri_mod{}".format(mod): "* - DEBUG    - Module {0} HV ".format(mod) for mod in mod_config['module_id']}
    blocksHV = extract_values(
                filepath='/data/wipac/CTA/targetcdata/run{0}_log.log'.format(run),
                block_start_pattern="* - main            - INFO     - Run {0} Sub-Run {1} has * events, total so far in run is *".format(run, int(subrun)-1) if int(subrun) > 0 else "* - main            - INFO     - Run Number: {0}".format(run),  # '*' matches the block number
                value_patterns=patternsHV,
            )

    hv = [blocksHV[0][key][0] for ind, key in enumerate(blocksHV[0].keys()) if ind != 0 and len(blocksHV[0][key]) > 0]

    patternsCur = {"Pri_mod{}".format(mod): "* - DEBUG    - Module {0} HV *V, ".format(mod) for mod in mod_config['module_id']}
    blocksCur = extract_values(
                filepath='/data/wipac/CTA/targetcdata/run{0}_log.log'.format(run),
                block_start_pattern="* - main            - INFO     - Run {0} Sub-Run {1} has * events, total so far in run is *".format(run, int(subrun)-1) if int(subrun) > 0 else "* - main            - INFO     - Run Number: {0}".format(run),  # '*' matches the block number
                value_patterns=patternsCur,
            )

    current = [blocksCur[0][key][0] for ind, key in enumerate(blocksCur[0].keys()) if ind != 0 and len(blocksCur[0][key]) > 0]
    try:
        np.save(fpm_hv_file.format(run, subrun), hv)
    except:
        pass
    
    try:
        np.save(fee_current_file.format(run, subrun), current)
    except:
        pass

    
for subrun in range(0, end_subrun+1):
    time.sleep(time_per_subrun)
    print(f"Generating... Run: {run}, Subrun: {subrun}")
    os.system(f'cp {r0_filename.format(run, subrun)} {outfolder_r0}')
    generate_temps_FPMs(run, subrun)
    generate_temps_FEEs(run, subrun)
    generate_HV_current(run, subrun)

print("END OF SIMULATION")
