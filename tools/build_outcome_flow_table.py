"""Render the final cohort flow as an English-status three-rule table.

Only presentation labels change. Counts are read from the audited S1 source.
Requires pdflatex (booktabs, tabularx, newtxtext/newtxmath).
"""
import argparse
import csv
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = r'''\documentclass[10pt]{article}
\usepackage[landscape,a4paper,margin=20mm]{geometry}
\usepackage{booktabs,array,tabularx}
\usepackage{newtxtext,newtxmath}
\pagestyle{empty}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}p{24mm}}
\begin{document}
\begin{center}
{\normalsize TABLE S1\par}
\vspace{2pt}
{\small\scshape Outcome classification by interview year and prediction horizon. Counts are mutually exclusive at the main category level. Indented rows are subsets, not additional participants. Death by the final wave excludes a participant regardless of prior dementia. Positive and negative correspond to outcome Y equal to 1 and 0, respectively.\par}
\end{center}
\vspace{6pt}
\footnotesize
\renewcommand{\arraystretch}{1.25}
\setlength{\tabcolsep}{5pt}
\noindent\begin{tabularx}{\textwidth}{@{}L >{\raggedright\arraybackslash}p{30mm} R R R R@{}}
\toprule
\textbf{Classification / outcome rule} & \textbf{Outcome status} & \textbf{2000: 6-year} \newline \textbf{(2000--2006)} & \textbf{2000: 8-year} \newline \textbf{(2000--2008)} & \textbf{2010: 6-year} \newline \textbf{(2010--2016)} & \textbf{2010: 8-year} \newline \textbf{(2010--2018)} \\
\midrule
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_source=ROOT/'artifacts/tables/supplementary/S1.csv'
    if not default_source.is_file():default_source=ROOT/'delivery/tables/audited_sources/supplementary/S1.csv'
    parser.add_argument('--source', type=Path, default=default_source)
    parser.add_argument('--output', type=Path, default=ROOT/'delivery/tables/outcome_flow')
    parser.add_argument('--tex-only', action='store_true')
    args = parser.parse_args()
    rows = list(csv.DictReader(args.source.open(encoding='utf-8-sig')))
    windows = ['2000-2006', '2000-2008', '2010-2016', '2010-2018']
    by = {r['Category'].strip(): r for r in rows}
    for w in windows:
        assert int(by['Analytic cohort'][w]) == int(by['Incident dementia'][w]) + int(by['Confirmed non-dementia at the final wave'][w])
        excluded = sum(int(r[w]) for r in rows if r['Classification']=='Excluded' and r['Row type']!='subset')
        assert int(by['Eligible at baseline'][w]) == int(by['Analytic cohort'][w]) + excluded
        assert int(by['Died by the final wave'][w]) == int(by['With observed dementia'][w]) + int(by['Without observed dementia'][w])
    args.output.mkdir(parents=True,exist_ok=True)
    lines=[]
    for r in rows:
        name = r['Category'].strip()
        status = {'Y = 1':'Positive','Y = 0':'Negative','Excluded':'Excluded'}.get(r['Classification'])
        if status is None: status = 'Baseline' if name=='Eligible at baseline' else 'Positive or negative'
        r['Classification']=status
        label = (r'\hspace*{1em}' if r['Row type']=='subset' else '') + name
        lines.append(' & '.join([label,status]+[f'{int(r[w]):,}' for w in windows])+r' \\')
    stem=args.output/'Supplementary_Table_S1_English_status'
    stem.with_suffix('.tex').write_text(PREAMBLE+'\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabularx}'+'\n'+r'\end{document}'+'\n',encoding='utf-8')
    with stem.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    if not args.tex_only:
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={args.output}',str(stem.with_suffix('.tex'))],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        for ext in ['.aux','.log']:stem.with_suffix(ext).unlink(missing_ok=True)
    print(stem.with_suffix('.pdf'))

if __name__=='__main__': main()
