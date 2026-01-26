#!/usr/bin/env python3
"""
Parse E_p ladder log files and extract parameters and results into a table.

Filename format: E_p_ladder_L_{L}_U_{U}_V_{V}_t0_{t0}_density_{density}.log
"""

import re
import sys
import csv
from pathlib import Path

def parse_value(s):
    """Convert string to float, handling both underscore and decimal formats."""
    if '_' in s and '.' not in s:
        return float(s.replace('_', '.'))
    else:
        return float(s)

def parse_filename(filename):
    """Extract parameters from filename. Returns dict or None if no match."""
    # Skip offset files
    if 'offset' in filename:
        return None
    
    # Pattern: E_p_ladder_L_{L}_U_{U}_V_{V}_t0_{t0}_density_{density}[_chi_{maxdim}].log
    num = r'(-?\d+(?:[._]\d+)?)'
    pattern = rf'^E_p_ladder_L_(\d+)_U_{num}_V_{num}_t0_{num}_density_{num}(?:_chi_(\d+))?\.log$'
    
    match = re.match(pattern, filename)
    if not match:
        return None
    
    g = match.groups()
    return {
        'L': int(g[0]),
        'U': parse_value(g[1]),
        'V': parse_value(g[2]),
        't0': parse_value(g[3]),
        'density': parse_value(g[4]),
        'chi': int(g[5]) if g[5] is not None else 1000,
    }

def extract_Ep_and_convergence(filepath):
    """Extract E_p, E(N), and relative energy convergence from log file.
    
    Returns (E_p, E_N, rel_diff) where rel_diff is |E_final - E_penultimate| / |E_final| for E(N) calculation.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract E_p
        E_p = None
        match = re.search(r'RESULT:\s*E_p\s*=\s*([-\d.eE+]+)', content)
        if match:
            E_p = float(match.group(1))
        else:
            match = re.search(r'E_p\s*=\s*([-\d.eE+]+)\s*\(', content)
            if match:
                E_p = float(match.group(1))
        
        # Extract E(N)
        E_N = None
        match = re.search(r'E\(N\)\s*=\s*([-\d.eE+]+)', content)
        if match:
            E_N = float(match.group(1))
        
        # Extract relative energy difference from E(N) calculation (section [3/3])
        rel_diff = None
        
        # Find the [3/3] section
        section_match = re.search(r'\[3/3\] Computing E\(N.*?\n(.*?)(?:E\(N\)\s*=|$)', content, re.DOTALL)
        if section_match:
            section = section_match.group(1)
            
            # Find all sweep energies in this section
            sweep_matches = re.findall(r'After sweep (\d+) energy=([-\d.eE+]+)', section)
            
            if len(sweep_matches) >= 2:
                # Get the last two sweeps
                penultimate_sweep, E_penultimate = sweep_matches[-2]
                final_sweep, E_final = sweep_matches[-1]
                E_penultimate = float(E_penultimate)
                E_final = float(E_final)
                rel_diff = abs(E_final - E_penultimate) / abs(E_final)
        
        return E_p, E_N, rel_diff
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None, None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_Ep_logs.py <directory> [output.csv]")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    path = Path(directory)
    log_files = list(path.glob('E_p_ladder_*.log'))
    print(f"Found {len(log_files)} E_p_ladder log files in {directory}")
    
    results = []
    for filepath in sorted(log_files):
        params = parse_filename(filepath.name)
        if params is None:
            continue
        
        E_p, E_N, rel_diff = extract_Ep_and_convergence(filepath)
        if E_p is None:
            print(f"  Warning: Could not extract E_p from {filepath.name}", file=sys.stderr)
            continue
        
        # Filter out entries with poor convergence
        if rel_diff is None or rel_diff > 1e-6:
            print(f"  Skipping (rel_diff={rel_diff:.2e} > 1e-6): {filepath.name}" if rel_diff else f"  Skipping (no convergence data): {filepath.name}", file=sys.stderr)
            continue
        
        params['E_p'] = E_p
        params['E_N'] = E_N
        params['rel_diff'] = rel_diff
        results.append(params)
    
    if not results:
        print("No valid results found!")
        sys.exit(1)
    
    results.sort(key=lambda x: (x['L'], x['U'], x['V'], x['t0'], x['density'], x['chi']))
    
    columns = ['L', 'U', 'V', 't0', 'density', 'chi', 'E_N', 'E_p', 'rel_diff']
    
    # Print table
    print("\n" + "=" * 130)
    print("RESULTS TABLE")
    print("=" * 130)
    header = "  ".join(f"{col:>12}" for col in columns)
    print(header)
    print("-" * len(header))
    
    for r in results:
        rel_diff_str = f"{r['rel_diff']:.2e}" if r['rel_diff'] is not None else "N/A"
        E_N_str = f"{r['E_N']:.8f}" if r['E_N'] is not None else "N/A"
        row = f"{r['L']:>12}  {r['U']:>12.4f}  {r['V']:>12.4f}  {r['t0']:>12.4f}  {r['density']:>12.6f}  {r['chi']:>12}  {E_N_str:>12}  {r['E_p']:>12.8f}  {rel_diff_str:>12}"
        print(row)
    
    print(f"\nTotal: {len(results)} files processed")
    
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in results:
                row_dict = {col: r[col] if r.get(col) is not None else '' for col in columns}
                writer.writerow(row_dict)
        print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
