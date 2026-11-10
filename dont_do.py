"""HRS "don't do" answers for functional-limitation items.

RAND HRS codes the "any difficulty" items (RwWALKSA, RwMAPA, ...) as 0/1 and sets the core-interview answer
"7. don't do" to missing. Those missing values are not random: people who no longer do an activity have a
higher dementia risk than people who report difficulty. This module reads the HRS core
answers so "don't do" can be kept as its own category.
"""
import re
import pandas as pd

# RAND item (without wave prefix) -> HRS core variable, per wave.
# Wave 5 (2000): module E, file H00E_R.  Wave 10 (2010): module G, file H10G_R.
CORE_ITEMS = {
    5: ('h00da/H00E_R.DA', 'h00sta/H00E_R.DCT', {
        'walksa': 'G2689', 'walk1a': 'G2692', 'sita': 'G2695', 'chaira': 'G2698', 'climsa': 'G2701',
        'clim1a': 'G2704', 'stoopa': 'G2707', 'armsa': 'G2710', 'pusha': 'G2713', 'lifta': 'G2716',
        'dimea': 'G2719', 'toilta': 'G2775', 'mapa': 'G2851'}),
    10: ('h10da/H10G_R.da', 'h10sta/H10G_R.dct', {
        'walksa': 'MG001', 'walk1a': 'MG003', 'sita': 'MG004', 'chaira': 'MG005', 'climsa': 'MG006',
        'clim1a': 'MG007', 'stoopa': 'MG008', 'armsa': 'MG009', 'pusha': 'MG010', 'lifta': 'MG011',
        'dimea': 'MG012', 'toilta': 'MG030', 'mapa': 'MG040'}),
}

DONT_DO = 7


def read_core_answers(wave, original_dir='data/original data'):
    """Return a DataFrame with hhidpn and the raw core answer for each functional item (columns = RAND item names)."""
    da, dct, items = CORE_ITEMS[wave]
    with open(f'{original_dir}/{dct}') as f:
        dictionary = f.read()
    spans = {}
    for m in re.finditer(r'_column\((\d+)\)\s+\S+\s+(\w+)\s+%(\d+)', dictionary):
        start, name, width = int(m.group(1)), m.group(2), int(m.group(3))
        spans[name] = (start - 1, start - 1 + width)
    cols = ['HHID', 'PN'] + list(items.values())
    core = pd.read_fwf(f'{original_dir}/{da}', colspecs=[spans[c] for c in cols], names=cols, dtype=str)
    out = pd.DataFrame({'hhidpn': (core['HHID'] + core['PN']).astype(int)})
    for item, var in items.items():
        out[item] = pd.to_numeric(core[var], errors='coerce')
    return out
