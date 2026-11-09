"""Build and validate the unified catalog for both cohort processing stages."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
MISSING = ['', 'NaN', 'nan']
CONDITIONS = {'hibpe':'high blood pressure', 'diabe':'diabetes', 'cancre':'cancer',
              'lunge':'lung disease', 'hearte':'heart problems', 'stroke':'stroke', 'arthre':'arthritis'}
FUNCTIONS = {'walksa':'walking several blocks', 'walk1a':'walking one block', 'sita':'sitting two hours',
             'chaira':'getting up from a chair', 'climsa':'climbing several flights of stairs',
             'clim1a':'climbing one flight of stairs', 'lifta':'lifting or carrying 10 pounds',
             'stoopa':'stooping, kneeling, or crouching', 'dimea':'picking up a dime',
             'armsa':'raising arms above shoulder level', 'pusha':'pushing or pulling large objects'}
CESD = {'depres':'feeling depressed', 'effort':'everything being an effort', 'sleepr':'restless sleep',
        'flone':'feeling lonely', 'fsad':'feeling sad', 'going':'difficulty getting going'}
LEVELS = {'education4':{'1':'0-7 years','2':'8-11 years','3':'12 years','4':'13 or more years'},
          'race3':{'1':'Hispanic','2':'non-Hispanic Black','3':'non-Black/non-Hispanic'},
          'region4':{'1':'Northeast','2':'North Central/Midwest','3':'South','4':'West'},
          'relationship4':{'1':'married or partnered','2':'separated or divorced','3':'widowed','4':'never married'}}
LABELS = {'age_years':'Age at baseline interview in completed years', 'female':'Recorded sex, 0 male and 1 female',
          'height_m':'Self-reported height in meters', 'weight_kg':'Self-reported weight in kilograms',
          'bmi':'Body mass index, weight_kg divided by height_m squared',
          'self_memory_0_4':'Self-rated memory, 0 excellent to 4 poor',
          'shlt_0_4':'Self-rated general health, 0 excellent to 4 poor',
          'employed':'Employment at the baseline interview', 'smoke_ever100':'Ever smoked at least 100 cigarettes',
          'smoke_now':'Current smoking', 'health_limits_work':'Health limits ability to work',
          'vigorous_high_approx':'Approximate harmonized high vigorous-activity indicator',
          'education4':'Completed schooling grouped into four categories', 'race3':'Harmonized three-category race/ethnicity',
          'region4':'Harmonized four-region residence classification', 'relationship4':'Harmonized four-category relationship status',
          'afqt_3':'AFQT score from the 1980 assessment, stored source field divided by 1000'}
SPECIAL = {
 'cohort':('Cohort identifier, HRS or NLSY79','identifier','text'),
 'person_id':('Cohort-prefixed person identifier','identifier','text'),
 'household_id':('Cohort-prefixed household grouping identifier','group','text'),
 'pair_id':('Person identifier and baseline/outcome survey years defining one pair','identifier','text'),
 'cv_fold5':('Fixed household hash assignment to fold 0 through 4','split','category'),
 'analysis_role':('Development or locked_test, with fold 4 assigned to locked_test','split','category'),
 'followup_band':('Reporting interval [4,8), [8,12), or [12,16] actual years','reporting','category'),
 'is_first_eligible_pair':('Earliest eligible baseline then earliest eligible follow-up per person','selection','binary'),
 'baseline_cognition27':('Baseline cognition total, immediate + delayed + serial7 + backward20, range 0-27','candidate_predictor','score'),
 'outcome_cognition27':('Follow-up cognition total, range 0-27','outcome','score'),
 'followup_years':('Elapsed interview months divided by 12, retaining fractional years','candidate_predictor','continuous'),
 'baseline_sampling_weight':('Positive provider cross-sectional respondent weight at baseline','weight','continuous'),
 'baseline_telephone':('Baseline interview was by telephone, 1 yes and 0 face-to-face','candidate_predictor','binary'),
 'birth_year':('Recorded calendar year of birth','candidate_predictor','integer'),
 'income_asinh2020k_approx':('Approximate cross-cohort income, asinh(2020 dollars / 1000)','sensitivity_predictor','continuous'),
 'cognition27_strict_observed':('Observed four-component sum for eligible self-report telephone/face-to-face assessments','cognition','score'),
 'observed_component_sum27':('Sum of four observed components when all are available','cognition','score'),
 'official_cog27':('Provider CogImp cognition total with provider imputation and mode adjustment','cognition_sensitivity','score'),
 'official_component_sum27_unadjusted':('Sum of four official components before provider mode adjustment','cognition_sensitivity','score'),
 'rand_published_cog27':('Cognition total as published in the RAND release','cognition_reference','score'),
 'strict27_eligible':('Self-report, telephone/face-to-face cognition, all four observed components and all imputation flags zero','selection','binary'),
 'baseline3_eligible':('Self-report telephone/face-to-face assessment with observed immediate recall, serial7 and backward20','selection','binary'),
 'baseline3_sum17_observed':('Immediate recall + serial7 + backward20, requiring all three components, range 0-17','cognition','score'),
 'official27_eligible':('Self-report with official CogImp total and a recognized cognition mode','selection','binary'),
 'cognition_mode':('CogImp mode, 1 telephone/face-to-face and 2 web','assessment_metadata','category'),
 'cognition_mode_rand':('Cognition mode code preserved from RAND','assessment_metadata','category'),
 'cogimp_status':('CogImp respondent assessment status used to identify self reports','assessment_metadata','category'),
 'any_component_imputed':('Any official component flagged as imputed, missing when any flag is unavailable','quality','binary'),
 'proxy_rand':('Proxy interview flag from RAND','assessment_metadata','category'),
 'proxy_tracker':('Proxy interview flag from Tracker','assessment_metadata','category'),
 'interview_mode_tracker':('Tracker interview mode, 1 face-to-face, 2 telephone, 3 web','assessment_metadata','category'),
 'interview_month_index':('Interview ending calendar year times 12 plus month minus 1','timing','integer'),
 'birth_date_flag':('RAND birth-date source/quality flag','quality','category'),
 'rand_mid_date_flag':('RAND interview midpoint-date flag','quality','category'),
 'rand_interview_status':('RAND interview status, retained interviews have status 1 or 2','assessment_metadata','category'),
 'age_months_midpoint':('RAND age at interview midpoint in months','timing','continuous'),
 'reported_bmi_rand':('BMI reported by RAND for comparison with the project-derived BMI','quality','continuous'),
 'survey_stratum':('Provider sampling stratum','survey_design','category'),
 'survey_secu':('Provider sampling error computation unit','survey_design','category'),
 'household_income_nominal':('RAND respondent-and-spouse income in nominal dollars','auxiliary','continuous'),
 'household_wealth_nominal':('RAND household wealth in nominal dollars','auxiliary','continuous'),
}


def describe(field, rules):
    if field in SPECIAL:
        label, role, dtype = SPECIAL[field]
        return {'description':label, 'role':role, 'data_type':dtype}
    if field.endswith('__source_year'):
        base = field.removesuffix('__source_year')
        return {'description':f'Source survey year (or recorded aggregate bound) for {base}', 'role':'provenance', 'data_type':'integer'}
    if '__' in field:
        base, level = field.rsplit('__',1)
        parent = describe(base, rules)
        category = parent.get('category_labels',{}).get(level, level)
        return dict(parent, description=f'Indicator that {parent["description"]} is {category}',
                    data_type='binary', coding=f'1 category {level}; 0 another observed category; blank when parent is missing', parent_field=base)
    if field in rules['nlsy'] or field in rules['common']:
        entry = rules['nlsy'].get(field, {})
        common = rules['common'].get(field,{})
        core = rules['core'].get(field,{})
        label = LABELS.get(field, field.replace('_',' ').capitalize())
        if field in CONDITIONS: label=f'Ever diagnosed with {CONDITIONS[field]}'
        if field.startswith('cesd_') and field[5:] in CESD: label='Shared depression symptom, '+CESD[field[5:]]
        if field.startswith('function_'):
            label='Functional difficulty with '+FUNCTIONS[field[len('function_'):-1]]
        categorical = field in rules['categories'] or '_raw' in field
        dtype = 'category' if categorical else 'numeric'
        result = {'description':label,'role':'candidate_predictor' if common else 'extension_or_auxiliary',
                  'data_type':dtype,'nlsy_source':core.get('nlsy_source',entry.get('source','')),
                  'hrs_source':common.get('hrs_official_source_rule',''),
                  'nlsy_derivation':core.get('coding_and_limits',entry.get('rule','')),
                  'harmonization':common.get('status','NLSY79-specific'),
                  'source_time':'Baseline or earlier; consult the per-person source-year table for NLSY79',
                  'coding':core.get('coding_and_limits',entry.get('rule',''))}
        if field in LEVELS: result['category_labels']=LEVELS[field]
        if field.startswith('function_'):
            result['category_labels']={'0':'no difficulty','1':'difficulty or cannot do','2':'does not do'}
        if field.startswith('health_history_'):
            parent=field.removeprefix('health_history_')
            result['description']='Latest eligible historical measurement of '+describe(parent,rules)['description']
        if field.startswith('education4'):
            result['source_time']='HRS RAEDYRS uses the first nonmissing report across waves with Tracker fallback; per-person source wave is absent from this export. NLSY79 uses baseline-round schooling.'
        return result
    for component, limit in [('immediate',10),('delayed',10),('serial7',5),('backward20',2)]:
        if field in [f'baseline_{component}',f'baseline_{component}_strict']:
            return {'description':f'Baseline {component} cognition component, range 0-{limit}', 'role':'sensitivity_predictor','data_type':'score'}
        if field.startswith(component+'_'):
            suffix=field[len(component)+1:]
            return {'description':f'{component} cognition component (0-{limit}), {suffix.replace("_"," ")}',
                    'role':'provenance' if suffix=='observed_source' else 'cognition',
                    'data_type':'text' if suffix=='observed_source' else 'binary' if suffix=='imputation_flag' else 'score'}
    aliases={'baseline_total27_strict':'baseline_cognition27','baseline_age_years':'age_years',
             'baseline_source_year':'baseline_survey_year','followup_years_from_interview_months':'followup_years',
             'outcome2020_total27_strict':'outcome_cognition27','outcome2020_interview_year':'outcome_interview_year',
             'outcome2020_interview_month':'outcome_interview_month'}
    if field in aliases: return dict(describe(aliases[field],rules), alias_of=aliases[field])
    if field.startswith('baseline_') and field.removeprefix('baseline_') in SPECIAL:
        result=describe(field.removeprefix('baseline_'),rules)
        return dict(result,description='Baseline '+result['description'])
    if field.startswith('outcome_') and field.removeprefix('outcome_') in SPECIAL:
        result=describe(field.removeprefix('outcome_'),rules)
        return dict(result,description='Follow-up '+result['description'],role='outcome_metadata')
    if field.endswith(('_survey_year','interview_year','reference_year')) or field in ['survey_year','interview_year']:
        return {'description':field.replace('_',' ').capitalize()+' as a calendar year', 'role':'timing','data_type':'integer'}
    if field.endswith('interview_month') or field=='interview_month':
        return {'description':field.replace('_',' ').capitalize()+', 1-12','role':'timing','data_type':'integer'}
    if field.startswith('weight_') or field.endswith('_sampling_weight_raw'):
        return {'description':field.replace('_',' ').capitalize()+'; provider survey weight', 'role':'weight','data_type':'continuous'}
    if field.endswith('_matched'):
        return {'description':'Person successfully linked to '+field.removesuffix('_matched'), 'role':'quality','data_type':'binary'}
    if field.endswith('_stata_day'):
        return {'description':field.replace('_',' ').capitalize()+'; days since 1960-01-01','role':'timing','data_type':'integer'}
    if field.startswith('tracker_') or field=='birth_month':
        return {'description':field.replace('_',' ').capitalize(), 'role':'timing','data_type':'integer'}
    if field in ['income_dollars2020','wealth_dollars2020','income_asinh2020k','wealth_asinh2020k']:
        return {'description':field.replace('_',' ').capitalize()+'; CPI-adjusted to 2020 dollars, asinh variants divide by 1000 before transformation', 'role':'auxiliary','data_type':'continuous'}
    if field in rules['supplemental']:
        return rules['supplemental'][field]
    raise ValueError('Missing field definition: '+field)


def iter_table(payload, name):
    data=gzip.decompress(payload) if name.endswith('.gz') else payload
    return csv.DictReader(io.StringIO(data.decode('utf-8-sig'),newline=''))


def enrich(field, definition, rules):
    result=dict(definition)
    base=field.split('__')[0]
    sources={
        'birth_year':('RABYEAR','R0000500'),
        'baseline_telephone':('Tracker wave IWMODE, 1 -> 0 and 2 -> 1','INTMODE, 71/74 -> 0 and 72/75 -> 1'),
        'baseline_sampling_weight':('Positive wave WGTR from Tracker Weights','Positive SAMPWEIGHT at baseline'),
        'followup_years':('RAND interview ending year/month at both assessments','CURDATE year/month for cognition source round and 2020 round'),
        'baseline_cognition27':('Sum of IMRC + DLRC + SER7 + BWC20 with strict flags','XRND cognition module at COGNITION_SOURCEYR; sum of four audited components'),
        'outcome_cognition27':('Follow-up sum of IMRC + DLRC + SER7 + BWC20 with strict flags','2020 cognition module; sum of four audited components'),
        'baseline_immediate':('IMRC, range 0-10','COGNITION_4G list totals, range 0-10'),
        'baseline_delayed':('DLRC, range 0-10','COGNITION_8G list totals, range 0-10'),
        'baseline_serial7':('SER7, range 0-5','COGNITION_7A through 7E; each subtraction scored from previous actual answer'),
        'baseline_backward20':('BWC20, range 0-2','COGNITION_6C and 6F, first/second attempt scoring'),
        'income_asinh2020k_approx':('RAND respondent-plus-spouse income and CPI-U','Same-round TNFI_TRUNC family income and CPI-U'),
    }
    if base in sources:result.update(zip(['hrs_source','nlsy_source'],sources[base]))
    result.setdefault('missing_rule','Blank cell; preserve absence through joins and encoding. See source-specific derivation for skip handling.')
    if result['data_type'] in ['category','binary','text']:
        unit='category or code' if result['data_type']!='text' else 'text identifier or source label'
    elif result['data_type']=='score':unit='score points'
    elif 'source_year' in field or 'birth_year' in field or 'survey_year' in field or 'interview_year' in field:unit='calendar year'
    elif field.endswith('_stata_day'):unit='days since 1960-01-01'
    elif base=='height_m':unit='meters'
    elif base=='weight_kg':unit='kilograms'
    elif base in ['bmi','reported_bmi_rand']:unit='kg/m^2'
    elif 'asinh' in base:unit='asinh(2020 USD / 1000)' + (' per year' if 'slope' in base else '')
    elif 'dollars2020' in base:unit='2020 USD'
    elif 'nominal' in base and ('income' in base or 'wealth' in base):unit='nominal USD'
    elif 'weeks' in base and 'n_observed' not in base:unit='weeks'
    elif 'percent' in base:unit='percent'
    elif 'month' in base:unit='months or calendar month as specified'
    elif base in ['age_years','baseline_age_years','outcome_age_years','followup_years','nominal_followup_years']:unit='years'
    elif base=='afqt_3':unit='AFQT source score / 1000'
    else:unit='source units or counts as specified by the derivation'
    result['unit']=unit
    result['category_codebook_keys']=[k for k,v in rules['codebook_categories'].items()
                                      if v['question'] in result.get('nlsy_source','')]
    return result


def build(stage1, stage2, output):
    rules=json.loads((ROOT/'metadata/source_rules.json').read_text())
    config=json.loads((ROOT/'stage2_config.json').read_text())
    features=config['features']['feature_sets']
    tables=[]; fields={}; model_rows=[]; source_years={}; nlsy_baselines={}; model_summary={}
    for stage, path in [('01_harmonized',stage1),('02_model_ready',stage2)]:
        with ZipFile(path) as archive:
            names=[n for n in archive.namelist() if n.startswith('data/processed/hrs_complete_v3/')]
            if stage=='02_model_ready':names=['HRS.csv.gz','NLSY79.csv.gz']
            for name in names:
                payload=archive.read(name); reader=iter_table(payload,name); columns=reader.fieldnames
                cohort='HRS' if Path(name).name.lower().startswith('hrs') else 'NLSY79'
                key=['pair_id'] if 'pair_id' in columns else ['person_id','survey_year'] if 'survey_year' in columns and cohort=='HRS' else ['person_id']
                missing=dict.fromkeys(columns,0); count=0; people=set(); keys=set(); households={}; roles={}
                if stage=='01_harmonized':
                    assert hashlib.sha256(payload).hexdigest()==rules['stage1_table_sha256'][Path(name).name], 'Stage 01 content changed: '+name
                else:
                    assert hashlib.sha256(gzip.decompress(payload)).hexdigest()==config['output_csv_sha256'][name], 'Model table content changed: '+name
                    required=features['common_expanded77']+(features['nlsy_additional82'] if cohort=='NLSY79' else [])
                    assert set(required)<=set(columns)
                for row in reader:
                    count+=1; people.add(row.get('person_id',''))
                    identity=tuple(row[c] for c in key)
                    assert all(value not in MISSING for value in identity) and identity not in keys, 'Invalid primary key: '+name
                    keys.add(identity)
                    for column in columns: missing[column]+=row[column] in MISSING
                    if name.endswith('nlsy_feature_measurement_years.csv.gz'):source_years[row['person_id']]=row
                    if stage=='02_model_ready':
                        household=row['household_id']; fold=int(float(row['cv_fold5']))
                        assert fold==int(hashlib.sha256(('Part2-v2|'+household).encode()).hexdigest()[:12],16)%5
                        assert households.setdefault(household,fold)==fold
                        for column in columns:
                            if '__' in column and row[column] not in MISSING:
                                assert float(row[column]) in (0,1)
                        assert 48<=float(row['age_years'])<=55 and 4<=float(row['followup_years'])<=16
                        assert 0<=float(row['outcome_cognition27'])<=27
                        assert sum(float(row['baseline_'+c]) for c in ['immediate','delayed','serial7','backward20'])==float(row['baseline_cognition27'])
                        duration=float(row['followup_years'])
                        assert row['followup_band']==('4_to_lt8' if duration<8 else '8_to_lt12' if duration<12 else '12_to_16')
                        if cohort=='NLSY79':
                            nlsy_baselines[row['person_id']]=float(row['baseline_survey_year'])
                            assert row['analysis_role']==('locked_test' if fold==4 else 'development')
                            roles[row['analysis_role']]=roles.get(row['analysis_role'],0)+1
                table={'id':stage+'/'+Path(name).name,'stage':stage,'archive_member':name,'cohort':cohort,
                       'rows':count,'people':len(people),'primary_key':key,'columns':columns,
                       'sha256':hashlib.sha256(payload).hexdigest(),'missing_counts':missing,
                       'primary_key_validated':True,
                       'cognition_context':('Official imputed/mode-adjusted HRS scores' if cohort=='HRS' else 'Strict observed NLSY79 scores paired with the HRS official-score sensitivity analysis') if 'official27_sensitivity' in name else 'Strict observed scores where explicitly labeled strict; source tables also contain reference and official variants'}
                tables.append(table)
                if stage=='02_model_ready':model_summary[cohort]={'rows':count,'people':len(people),'households':len(households),'roles':roles}
                for column in columns:
                    if column not in fields:fields[column]=enrich(column,describe(column,rules),rules)
                    if stage=='02_model_ready':
                        definition=dict(fields[column])
                        memberships=[n for n,cols in features.items() if column in cols and (cohort=='NLSY79' or n.startswith('common_'))]
                        model_rows.append({'dataset':cohort,'column':column,'description':definition['description'],
                                           'data_type':definition['data_type'],'unit':definition['unit'],'role':definition['role'],
                                           'feature_sets':'|'.join(memberships),'rows':count,'missing':missing[column],
                                           'source':definition.get('hrs_source' if cohort=='HRS' else 'nlsy_source','Derived or provider design field; see catalog'),
                                           'coding':definition.get('coding','See catalog definitions and global conventions'),
                                           'source_time':definition.get('source_time','Defined by baseline/outcome prefix or design role'),
                                           'harmonization':definition.get('harmonization','Defined by cohort and table context')})
    timing_checks=0
    for person,baseline in nlsy_baselines.items():
        for column in set(features['common_expanded77']+features['nlsy_additional82']):
            parent=column.split('__')[0]; year=source_years[person].get(parent+'__source_year','')
            if year not in MISSING:
                assert float(year)<=baseline, 'Source year after baseline'
                timing_checks+=1
    assert len(tables)==34 and len(model_rows)==288
    output.mkdir(parents=True,exist_ok=True)
    catalog={'version':'unified_metadata_v1','stages':{'01_harmonized':'32 complete harmonized tables','02_model_ready':'Two current analysis tables'},
             'conventions':rules['conventions'],'fields':fields,'tables':tables,'feature_sets':features,
             'source_rules':rules,'stage2_configuration':config['features']}
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8')
    with (output/'model_fields.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(model_rows[0]));writer.writeheader();writer.writerows(model_rows)
    report={'status':'PASS','stage1_tables':32,'stage2_tables':2,'unique_fields':len(fields),
            'model_table_columns':len(model_rows),'model_unique_fields':len({r['column'] for r in model_rows}),
            'missing_definitions':0,'nlsy_nonmissing_predictor_source_year_checks':timing_checks,
            'primary_keys_validated':34,'stage1_source_hashes_verified':32,'model_csv_hashes_verified':2,
            'household_folds_verified':True,'model_samples':model_summary,
            'nlsy_postbaseline_source_years':0,'current_scope':'48-55 baseline, 4-16 actual follow-up, strict27',
            'documented_source_limitation':'RAND RAEDYRS source wave is unavailable per person in this export',
            'models_fitted':False}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage1',type=Path);parser.add_argument('stage2',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'metadata')
    args=parser.parse_args();build(args.stage1,args.stage2,args.output)
