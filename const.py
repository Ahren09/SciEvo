ID = "id"

UPDATED = "updated"


ARXIV_CATEGORIES_CS = ['cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL',
                       'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL',
                       'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR',
                       'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
                       'cs.MA', 'cs.ML', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE',
                       'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO',
                       'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY']

ARXIV_CATEGORIES_ECON = ['econ.EM', 'econ.GN', 'econ.TH']

ARXIV_CATEGORIES_EESS = ['eess.AS', 'eess.IV', 'eess.SP', 'eess.SY']

ARXIV_CATEGORIES_MATH = ['math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA',
                         'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS',
                         'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT',
                         'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG',
                         'math.MP', 'math.NA', 'math.NT', 'math.OA', 'math.OC',
                         'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG',
                         'math.SP', 'math.ST']

ARXIV_CATEGORIES_PHYSICS = ['astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA',
                            'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR',
                            'cond-mat.dis-nn', 'cond-mat.mes-hall',
                            'cond-mat.mtrl-sci', 'cond-mat.other',
                            'cond-mat.quant-gas', 'cond-mat.soft',
                            'cond-mat.stat-mech', 'cond-mat.str-el',
                            'cond-mat.supr-con', 'gr-qc', 'hep-ex', 'hep-lat',
                            'hep-ph', 'hep-th', 'math-ph', 'nlin.AO', 'nlin.CD',
                            'nlin.CG', 'nlin.PS', 'nlin.SI', 'nucl-ex',
                            'nucl-th', 'physics.acc-ph', 'physics.ao-ph',
                            'physics.app-ph', 'physics.atm-clus',
                            'physics.atom-ph', 'physics.bio-ph',
                            'physics.chem-ph', 'physics.class-ph',
                            'physics.comp-ph', 'physics.data-an',
                            'physics.ed-ph', 'physics.flu-dyn',
                            'physics.gen-ph', 'physics.geo-ph',
                            'physics.hist-ph', 'physics.ins-det',
                            'physics.med-ph', 'physics.optics',
                            'physics.plasm-ph', 'physics.pop-ph',
                            'physics.soc-ph', 'physics.space-ph', 'quant-ph']

ARXIV_CATEGORIES_QBIO = ['q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN',
                         'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM',
                         'q-bio.SC', 'q-bio.TO']

ARXIV_CATEGORIES_QFIN = ['q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF',
                         'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST',
                         'q-fin.TR']

ARXIV_CATEGORIES_STAT = ['stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT',
                         'stat.TH']

ARXIV_SUBJECTS = {
    "cs": ARXIV_CATEGORIES_CS,
    "econ": ARXIV_CATEGORIES_ECON,
    "eess": ARXIV_CATEGORIES_EESS,
    "math": ARXIV_CATEGORIES_MATH,
    "physics": ARXIV_CATEGORIES_PHYSICS,
    "q-bio": ARXIV_CATEGORIES_QBIO,
    "q-fin": ARXIV_CATEGORIES_QFIN,
    "stat": ARXIV_CATEGORIES_STAT,
}

ARXIV_SUBJECTS_LIST = [tag for name, cat in ARXIV_SUBJECTS.items() for tag in cat]

