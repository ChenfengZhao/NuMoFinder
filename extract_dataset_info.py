import pandas as pd
import re
import numpy as np
from collections import defaultdict

def extrac_dataset_info(args, input_path, fn_cm=None):
    """extract pre-defined information (i.e. a csv file and dataset list) from input path

    Parameters
    ----------
    args : class
        Input configurate class
    input_path : str
        input path of to all the dataset
    fn_cm : str, optional
        the input customized mods csv file
    
    Returns
    --------
    cpd_note_list : list of tuple
        target list of tuple: (rst_tab["CompoundName"], rst_tab["BaseType"])
    note_cpd_list_dict : default of list
        target dict of cpd list (rst_tab["BaseType"] : rst_tab["CompoundName"] list)
    ms1_mass_dict : dict of float
        target dict of ms1 list in float (cpd : ms1_mass)
    ms2_mass_dict : dict of str
        target dict of ms2 list string divided by space (cpd : ms2_mass string)
    """

    known_mods_dict = {'AnalyteForm': ['Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Native', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl', 'Permethyl'], 'NucleosideType': ['RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'RNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA', 'DNA'], 'BaseType': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'I', 'I', 'I', 'I', 'Y', 'Y', 'Y', 'Y', 'Y', 'Q', 'Q', 'Q', 'Q', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'sU', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'yW', 'yW', 'yW', 'yW', 'yW', 'yW', 'yW', 'yW', 'yW', 'yW', 'C', 'U', 'Y', 'D', 'C', 'C', 'U', 'Y', 'U', 'Y', 'C', 'C', 'sU', 'U', 'D', 'A', 'I', 'C', 'C', 'C', 'U', 'C', 'sU', 's2U', 'U', 'A', 'A', 'I', 'I', 'U', 'G', 'C', 'C', 'C', 'U', 'C', 'sU', 'A', 'A', 'A', 'I', 'G', 'G', 'C', 'G', 'U', 'U', 'sU', 'U', 'G', 'A', 'A', 'G', 'G', 'G', 'U', 'U', 'sU', 'U', 'U', 'U', 'yW', 'G', 'G', 'A', 'G', 'G', 'U', 'U', 'sU', 'U', 'G', 'A', 'U', 'U', 'U', 'Y', 'U', 'U', 'sU', 'D', 'U', 'yW', 'A', 'U', 'C', 'U', 'Y', 'A', 'C', 'U', 'A', 'U', 'A', 'sU', 'U', 'A', 'G', 'A', 'yW', 'G', 'sU', 'A', 'A', 'yW', 'A', 'sU', 'yW', 'A', 'yW', 'A', 'sU', 'yW', 'yW', 'G', 'yW', 'G', 'C', 'A', 'G', 'T', 'G', 'G', 'G', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'G', 'G', 'C', 'A', 'G', 'T', 'G', 'G', 'G', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'G', 'G'], 'CompoundName': ['A', 'ac6A', 'Am', 'ct6A', 'f6A', 'g6A', 'hm6A', 'hn6A', 'ht6A', 'i6A', 'io6A', 'm6Am', 'm2_8A', 'm2A_ m8A', 'm6_6A', 'm6_6Am', 'm6A', 'm6t6A', 'ms2hn6A', 'ms2i6A', 'ms2io6A', 'ms2t6A', 'msms2i6A', 't6A', 'ac4C', 'ac4Cm', 'C', 'C+', 'ca5C', 'Cm', 'f5C', 'f5Cm', 'hm5C', 'hm5Cm', 'ho5C', 'k2C', 'm3C', 'm4_4C', 'm4_4Cm', 'm4C', 'm4Cm', 'm5C', 'm5Cm', 's2C', 'acp3D', 'D', 'm5D', 'm5Dm', 'G', 'Gm', 'm1G_ m2G', 'm1Gm_ m2Gm', 'm2_2_7G', 'm2_2G', 'm2_2Gm', 'm2_7G', 'm7G', 'I', 'Im', 'm1I', 'm1Im', 'acp3Y', 'm1acp3Y', 'm1Y_ m3Y', 'Y', 'Ym', 'gluQ', 'manQ_ galQ', 'oQ', 'Q', 'cmnm5ges2U', 'cmnm5s2U', 'ges2U', 'm5s2U', 'mcm5s2U_ cm5s2U', 'mnm5ges2U', 'mnm5s2U', 'ncm5s2U', 'nm5ges2U', 'nm5s2U', 'sU', 'sUm', 'chm5U', 'cmnm5U', 'cmnm5Um', 'ho5U', 'inm5U', 'inm5Um', 'mchm5U', 'mchm5Um', 'mcm5U', 'mcm5Um', 'mcmo5U', 'mcmo5Um', 'mnm5U', 'nchm5U', 'ncm5U', 'ncm5Um', 'nm5U', 'mUm', 'Um', 'mU', 'U', 'imG', 'imG-14', 'imG2', 'mimG', 'OHyW', 'OHyWy', 'yW', 'yW-58', 'yW-72', 'yW-86', 'C', 'U', 'Y', 'D', 'mC', 'Cm', 'mU', 'mY', 'Um', 'Ym', 's2C', 'ho5C', 'sU', 'ho5U', 'm5D', 'A', 'I', 'f5C', 'm4_4C', 'mCm', 'mUm', 'hm5C', 'm5s2U', 's2Um', 'mo5U', 'mA', 'Am', 'm1I', 'Im', 'cnm5U', 'G', 'ac4C', 'f5Cm', 'm4_4Cm', 'mnm5U', 'hm5Cm', 'nm5s2U', 'f6A', 'm6_6A', 'm6Am', 'm1Im', 'mG', 'Gm', 'ac4Cm', 'm7G', 'ncm5U', 'cm5U', 'mnm5s2U', 'se2U', 'preQ0tRNA', 'ac6A', 'm6_6Am', 'm2_2G', 'm1Gm_ m2Gm', 'm2_7G', 'ncm5Um', 'mcm5U', 'ncm5s2U', 'nchm5U', 'cm5s2U', 'chm5U_ cmo5U', 'imG-14', 'G+', 'm2_2Gm', 'ms2m6A', 'm2_2_7G', 'm2_7Gm', 'mcm5Um', 'cmnm5U', 'mcm5s2U', 'mchm5U_ mcmo5U', 'imG_ imG2', 'i6A', 'nm5se2U', 'inm5U', 'acp3U', 'acp3Y', 'cmnm4Um', 'mchm5Um_ mcmo5Um', 'cmnm5s2U', 'acp3D', 'mnm5se2U', 'mimG', 'io6A', 'inm5Um', 'C+', 'inm5s2U', 'm1acp3Y', 'g6A', 'k2C', 'tm5U', 'ms2i6A', 'cmnm5se2U', 'ct6A', 'ges2U', 'tm5s2U', 'ms2io6A', 'QtRNA', 't6A', 'yW-86', 'oQtRNA', 'nm5ges2u', 'm6t6A_ hn6A', 'ht6A', 'yW-72', 'ms2ct6A', 'mnm5ges2U', 'yW-58', 'ms2t6A', 'OHyWy', 'ms2hn6A', 'cmnm5ges2U', 'yW', 'OHyW', 'gluQtRNA', 'o2yW', 'manQtRNA_ galQtRNA', 'C', 'A', 'G', 'T', 'mG', '8oxG', 'CEG', 'mC', '5hmC', '5fC', '5caC', 'e3C', 'mA', '6hmA', 'ncm6A', 'm2A', 'mT', '5hmU', '5fU', 'putThy', 'diHT', '5caU', '5NeOmdU', 'ADG', 'm22G', 'C', 'A', 'G', 'T', 'mG', '8oxG', 'CEG', 'mC', '5hmC', '5fC', '5caC', 'e3C', 'mA', '6hmA', 'ncm6A', 'm2A', 'mT', '5hmU', '5fU', 'putThy', 'diHT', '5caU', '5NeOmdU', 'ADG', 'm22G'], 'MolecularFormula': ['C15H8D15N5O4', 'C16H11D12N5O5', 'C15H11D12N5O4', 'C20H13D15N6O7', 'C15H9D12N5O5', 'C19H10D18N6O7', 'C16H10D15N5O5', 'C23H15D21N6O8', 'C23H12D24N6O9', 'C19H17D12N5O4', 'C20H16D15N5O5', 'C15H14D9N5O4', 'C17H12D15N5O4', 'C16H10D15N5O4', 'C15H14D9N5O4', 'C15H17D6N5O4', 'C15H11D12N5O4', 'C22H16D18N6O8', 'C24H17D21N6O8S1', 'C20H19D12N5O4S1', 'C21H18D15N5O5S1', 'C23H15D21N6O8S1', 'C21H21D12N5O4S2', 'C22H13D21N6O8', 'C15H11D12N3O6', 'C15H14D9N3O6', 'C14H8D15N3O5', 'C23H16D27N7O4', 'C16H7D18N3O7', 'C14H11D12N3O5', 'C15H8D15N3O6', 'C15H11D12N3O6', 'C16H9D18N3O6', 'C16H12D15N3O6', 'C15H7D18N3O6', 'C23H17D24N5O6', 'C14H11D12N3O5', 'C14H14D9N3O5', 'C14H17D6N3O5', 'C14H11D12N3O5', 'C14H14D9N3O5', 'C15H10D15N3O5', 'C15H13D12N3O5', 'C14H8D15N3O4S1', 'C19H15D18N3O8', 'C15H10D18N2O7', 'C16H12D18N2O7', 'C16H15D15N2O7', 'C16H7D18N5O5', 'C16H10D15N5O5', 'C16H10D15N5O5', 'C16H13D12N5O5', 'C18H16D15N5O6', 'C16H13D12N5O5', 'C16H16D9N5O5', 'C18H13D18N5O6', 'C18H10D21N5O6', 'C14H8D12N4O5', 'C14H11D9N4O5', 'C14H11D9N4O5', 'C14H14D6N4O5', 'C20H12D21N3O8', 'C20H15D18N3O8', 'C14H10D12N2O6', 'C14H7D15N2O6', 'C14H10D12N2O6', 'C33H19D33N6O10', 'C35H21D36N5O12', 'C27H15D30N5O8', 'C26H14D27N5O7', 'C27H28D15N3O7', 'C18H11D18N3O7', 'C22H25D9N2O5', 'C14H10D12N2O5', 'C16H9D15N2O7', 'C25H29D12N3O5', 'C16H12D15N3O5', 'C17H9D18N3O6', 'C25H26D15N3O5', 'C16H9D18N3O5', 'C13H8D12N2O5S1', 'C13H11D9N2O5S1', 'C17H8D18N2O9', 'C19H11D18N3O9', 'C19H14D15N3O9', 'C14H7D15N2O7', 'C20H18D15N3O6', 'C20H21D12N3O6', 'C17H11D15N2O9', 'C17H14D12N2O9', 'C17H8D18N2O8', 'C17H11D15N2O8', 'C16H9D15N2O9', 'C16H12D12N2O9', 'C17H12D15N3O7', 'C18H8D21N3O8', 'C17H9D18N3O7', 'C17H12D15N3O7', 'C16H9D18N3O6', 'C13H14D6N2O6', 'C13H11D9N2O6', 'C13H11D9N2O6', 'C13H8D12N2O6', 'C17H14D9N5O5', 'C17H11D12N5O5', 'C18H13D12N5O5', 'C18H16D9N5O5', 'C26H17D21N6O10', 'C25H17D21N6O8', 'C25H24D12N6O9', 'C24H21D15N6O7', 'C24H18D18N6O7', 'C24H15D21N6O7', 'C9O5N3H13', 'C9O6N2H12', 'C9O6N2H12', 'C9O6N2H14', 'C10O5N3H15', 'C10O5N3H15', 'C10O6N2H14', 'C10O6N2H14', 'C10O6N2H14', 'C10O6N2H14', 'C9O4N3H13S1', 'C9H13N3O6', 'C9O5N2H12S1', 'C9O7N2H12', 'C10O6N2H16', 'C10O4N5H13', 'C10O5N4H12', 'C10O6N3H13', 'C11O5N3H17', 'C11O5N3H17', 'C11O6N2H16', 'C10O6N3H15', 'C10O5N2H14S1', 'C10O5N2H14S1', 'C10O7N2H14', 'C11O4N5H15', 'C11O4N5H15', 'C11O5N4H14', 'C11O5N4H14', 'C11H13N3O6', 'C10O5N5H13', 'C11O6N3H15', 'C11O6N3H15', 'C12O5N3H19', 'C11O6N3H17', 'C11O6N3H17', 'C10O5N3H15S1', 'C11H13N5O5', 'C12O4N5H17', 'C12O4N5H17', 'C12O5N4H16', 'C11O5N5H15', 'C11O5N5H15', 'C12O6N3H17', 'C11O5N5H17', 'C11O7N3H15', 'C11O8N2H14', 'C11O5N3H17S1', 'C9O5N2H12Se1', 'C12O5N5H13', 'C12O5N5H15', 'C13O4N5H19', 'C12O5N5H17', 'C12O5N5H17', 'C12O5N5H19', 'C12O7N3H17', 'C12O8N2H16', 'C11H15N3O6S', 'C11H15N3O8', 'C11O7N2H14S1', 'C11O9N2H14', 'C13O5N5H15', 'C12H16N6O5', 'C13O5N5H19', 'C12O4N5H17S1', 'C13O5N5H21', 'C13O5N5H21', 'C13O8N2H18', 'C12O8N3H17', 'C12O7N2H16S1', 'C12O9N2H16', 'C14O5N5H17', 'C15O4N5H21', 'C10O5N3H15Se1', 'C15H23N3O6', 'C13O8N3H19', 'C13O8N3H19', 'C13O8N3H19', 'C13H18N2O9', 'C12O7N3H17S1', 'C13H21N3O8', 'C11O5N3H17Se1', 'C15O5N5H19', 'C15O5N5H21', 'C16H25N3O6', 'C14H25N7O4', 'C15H23N3O5S', 'C14O8N3H21', 'C13O7N6H16', 'C15O6N5H25', 'C12O9N3H19S1', 'C16O4N5H23S1', 'C12O7N3H17Se1', 'C15O7N6H18', 'C19H28N2O5S', 'C12O8N3H19S2', 'C16O5N5H23S1', 'C17O7N5H23', 'C15O8N6H20', 'C17O7N6H22', 'C17O8N5H23', 'C20H31N3O5S', 'C16O8N6H22', 'C15H20N6O9', 'C18O7N6H24', 'C17H21N5O7S', 'C21H33N3O5S', 'C19O7N6H26', 'C16O8N6H22S1', 'C19H26N6O8', 'C17O8N6H24S1', 'C22H33N3O7S', 'C21O9N6H28', 'C21O10N6H28', 'C22O10N6H30', 'C21O11N6H28', 'C23O12N5H33', 'C9H13N3O4', 'C10H13N5O3', 'C10H13N5O4', 'C10H14N2O5', 'C11H15N5O4', 'C10H15N5O5', 'C13H17N5O6', 'C10H15N3O4', 'C10H15N3O5', 'C10H13N3O5', 'C10H13N3O6', 'C11H17N3O4', 'C11H17N5O3', 'C11H15N5O4', 'C12H16N6O4', 'C10H14N6O3', 'C11H16N2O5', 'C10H14N2O6', 'C10H12N2O6', 'C14H24N4O5', 'C9H14N2O5', 'C10H12N2O7', 'C12H19N3O6', 'C12H16N5O6', 'C12H18N5O5', 'C13H9D12N3O4', 'C14H9D12N5O3', 'C15H8D15N5O4', 'C13H11D9N2O5', 'C15H11D12N5O4', 'C16H7D18N5O5', 'C18H12D15N5O6', 'C14H11D12N3O4', 'C15H10D15N3O5', 'C14H9D12N3O5', 'C15H8D15N3O6', 'C15H13D12N3O4', 'C14H12D9N5O3', 'C15H11D12N5O4', 'C17H11D15N6O4', 'C16H8D18N6O3', 'C14H13D9N2O5', 'C13H10D12N2O5', 'C14H10D12N2O6', 'C20H18D18N4O5', 'C13H10D12N2O5', 'C14H8D12N3O6', 'C17H14D15N3O6', 'C20H7D24N5O6', 'C16H13D12N5O5'], 'Fragment': ['C7H3D6N5', 'C8H6D3N5O1', 'C7H3D6N5', 'C12H8D6N6O3', 'C8H5D6N5O1', 'C11H5D9N6O3', 'C8H5D6N5O1', 'C15H10D12N6O4', 'C14H7D15N6O5', 'C11H12D3N5', 'C12H11D6N5O1', 'C7H6D3N5', 'C9H7D6N5', 'C8H5D6N5', 'C7H11N5', 'C7H11N5', 'C7H6D3N5', 'C14H11D9N6O4', 'C16H12D12N6O4S1', 'C12H14D3N5S1', 'C13H13D6N5O1S1', 'C15H10D12N6O4S1', 'C13H16D3N5S2', 'C14H8D12N6O4', 'C7H6D3N3O2', 'C7H6D3N3O2', 'C6H3D6N3O1', 'C15H11D18N7', 'C8H2D9N3O3', 'C6H3D6N3O1', 'C7H3D6N3O2', 'C7H3D6N3O2', 'C8H4D9N3O2', 'C8H4D9N3O2', 'C7H2D9N3O2', 'C15H12D15N5O2', 'C6H6D3N3O1', 'C6H9N3O1', 'C6H9N3O1', 'C6H6D3N3O1', 'C6H6D3N3O1', 'C7H5D6N3O1', 'C7H5D6N3O1', 'C6H3D6N3S1', 'C11H10D9N3O4', 'C7H5D9N2O3', 'C8H7D9N2O3', 'C8H7D9N2O3', 'C8H2D9N5O1', 'C8H2D9N5O1', 'C8H5D6N5O1', 'C8H5D6N5O1', 'C10H11D6N5O2', 'C8H8D3N5O1', 'C8H8D3N5O1', 'C10H8D9N5O2', 'C10H5D12N5O2', 'C6H3D3N4O1', 'C6H3D3N4O1', 'C6H6N4O1', 'C6H6N4O1', 'C15H8D15N3O5', 'C15H11D12N3O5', 'C9H6D6N2O3', 'C9H3D9N2O3', 'C9H6D6N2O3', 'C25H14D24N6O6', 'C27H16D27N5O8', 'C19H10D21N5O4', 'C18H9D18N5O3', 'C19H23D6N3O3S1', 'C10H6D9N3O3S1', 'C14H20N2O1S1', 'C6H5D3N2O1S1', 'C8H4D6N2O3S1', 'C17H24D3N3O1S1', 'C8H7D6N3O1S1', 'C9H4D9N3O2S1', 'C17H21D6N3O1S1', 'C8H4D9N3O1S1', 'C5H3D3N2O1S1', 'C5H3D3N2O1S1', 'C9H3D9N2O5', 'C11H6D9N3O5', 'C11H6D9N3O5', 'C6H2D6N2O3', 'C12H13D6N3O2', 'C12H13D6N3O2', 'C9H6D6N2O5', 'C9H6D6N2O5', 'C9H3D9N2O4', 'C9H3D9N2O4', 'C8H4D6N2O5', 'C8H4D6N2O5', 'C9H7D6N3O3', 'C10H3D12N3O4', 'C9H4D9N3O3', 'C9H4D9N3O3', 'C8H4D9N3O2', 'C5H6N2O2', 'C5H3D3N2O2', 'C5H6N2O2', 'C5H3D3N2O2', 'C9H9N5O1', 'C9H6D3N5O1', 'C10H8D3N5O1', 'C10H11N5O1', 'C18H12D12N6O6', 'C17H12D12N6O4', 'C17H13D9N6O5', 'C16H16D6N6O3', 'C16H13D9N6O3', 'C16H10D12N6O3', 'C4O1N3H5', 'C4O2N2H4', 'C5O2N2H4', 'C4O2N2H6', 'C5O1N3H7', 'C4O1N3H5', 'C5O2N2H6', 'C6O2N2H6', 'C4O2N2H4', 'C5O2N2H4', 'C4N3H5S1', 'C4H5N3O2', 'C4O1N2H4S1', 'C4O3N2H4', 'C5O2N2H8', 'C5N5H5', 'C5O1N4H4', 'C5O2N3H5', 'C6O1N3H9', 'C5O1N3H7', 'C5O2N2H6', 'C5O2N3H7', 'C5O1N2H6S1', 'C4O1N2H4S1', 'C5O3N2H6', 'C6N5H7', 'C5N5H5', 'C6O1N4H6', 'C5O1N4H4', 'C6H5N3O2', 'C5O1N5H5', 'C6O2N3H7', 'C5O2N3H5', 'C6O1N3H9', 'C6O2N3H9', 'C5O2N3H7', 'C5O1N3H7S1', 'C6H5N5O1', 'C7N5H9', 'C6N5H7', 'C6O1N4H6', 'C6O1N5H7', 'C5O1N5H5', 'C6O2N3H7', 'C6O1N5H9', 'C6O3N3H7', 'C6O4N2H6', 'C6O1N3H9S1', 'C4O1N2H4Se1', 'C7O1N5H5', 'C7O1N5H7', 'C7N5H9', 'C7O1N5H9', 'C6O1N5H7', 'C7O1N5H11', 'C6O3N3H7', 'C7O4N2H8', 'C6H7N3O2S1', 'C6H7N3O4', 'C6O3N2H6S1', 'C6O5N2H6', 'C8O1N5H7', 'C7H8N6O1', 'C7O1N5H9', 'C7N5H9S1', 'C8O1N5H13', 'C7O1N5H11', 'C7O4N2H8', 'C7O4N3H9', 'C7O3N2H8S1', 'C7O5N2H8', 'C9O1N5H9', 'C10N5H13', 'C5O1N3H7Se1', 'C10N3O2H15', 'C8O4N3H11', 'C9O4N3H11', 'C7O4N3H9', 'C7O5N2H8', 'C7O3N3H9S1', 'C8H13N3O4', 'C6O1N3H9Se1', 'C10O1N5H11', 'C10O1N5H13', 'C10N3O2H15', 'C9H17N7', 'C10H15N3O1S1', 'C10O4N2H10', 'C8O3N6H8', 'C10O2N5H18', 'C7O5N3H11S1', 'C11N5H15S1', 'C7O3N3H9Se1', 'C10O3N6H10', 'C14H20N2O1S1', 'C7O4N3H11S2', 'C11O1N5H15S1', 'C12O3N5H15', 'C10O4N6H12', 'C12O3N6H14', 'C12O4N5H15', 'C15H23N3O1S1', 'C11O4N6H14', 'C10H12N6O5', 'C13O3N6H16', 'C12H13N5O3S1', 'C16H25N3O1S1', 'C14O3N6H18', 'C11O4N6H14S1', 'C14H18N6O4', 'C12O4N6H16S1', 'C17H25N3O3S1', 'C16O5N6H20', 'C16O6N6H20', 'C17O6N6H22', 'C16O7N6H20', 'C18O8N5H25', 'C4H5N3O1', 'C5H5N5', 'C5H5N5O1', 'C5H6N2O2', 'C6H7N5O1', 'C5H5N5O2', 'C8H9N5O3', 'C5H7N3O1', 'C5H7N3O2', 'C5H5N3O2', 'C5H5N3O3', 'C6H9N3O1', 'C6H9N5', 'C6H7N5O1', 'C7H8N6O1', 'C5H6N6', 'C6H8N2O2', 'C5H6N2O3', 'C5H4N2O3', 'C9H16N4O2', 'C5H8N2O2', 'C5H4N2O4', 'C7H11N3O3', 'C7H8N5O2', 'C7H10N5O1', 'C6H3D6N3O1', 'C7H3D6N5', 'C8H2D9N5O1', 'C6H5D3N2O2', 'C8H5D6N5O1', 'C9H1D12N5O2', 'C11H6D9N5O3', 'C7H5D6N3O', 'C8H4D9N3O2', 'C7H3D6N3O2', 'C8H2D9N3O3', 'C8H7D6N3O', 'C7H6D3N5', 'C8H5D6N5O', 'C10H5D9N6O', 'C9H2D12N6', 'C7H7D3N2O2', 'C6H4D6N2O2', 'C7H4D6N2O3', 'C13H12D12N4O2', 'C7H6D6N2O2', 'C7H2D6N3O3', 'C10H8D9N3O3', 'C12H2D15N5O2', 'C8H8D3N5O1']}
    
    # The dataframe of the known mods
    km_tab = pd.DataFrame(known_mods_dict)
    # km_tab = pd.read_csv(input_path + "/" + "KnownMods.csv")

    if fn_cm is not None:
        path_cm = input_path + "/" + fn_cm
        cm_tab = pd.read_csv(path_cm)

        # concat km_tab and cm_tab
        km_cm_tab = pd.concat([km_tab, cm_tab], ignore_index = True)
        km_cm_tab.reset_index()
    else:
        km_cm_tab = km_tab
    
    # if args.permethyl:
    #     analyte_form = "Permethyl"
    # else:
    #     analyte_form = "Native"
    analyte_form = args.analyte_form
    nucleoside_type = args.nucleoside_type

    rst_row_list = []
    for row in km_cm_tab.itertuples(index=False):
        if row.AnalyteForm == analyte_form and row.NucleosideType == nucleoside_type:
            rst_row_list.append(row)
    
    rst_tab = pd.DataFrame(rst_row_list)

    # print(rst_tab.duplicated())
    rst_tab.drop_duplicates(inplace=True, ignore_index=True)

    # dict of element mass
    mass_dict = {"C": 12.00000, "H": 1.00783, "O": 15.99491, "N": 14.00307, "S": 31.97207, "D": 2.01410, "Se": 79.91652}

    # calculate the basic ms1 mass for all the ms1 cpd
    ms1_mass_list = []
    for cpd in rst_tab["MolecularFormula"]:
        cpd_elem_list = []
        elem_weight_list = []
        match_mol = re.findall(r'([A-Za-z]+)(\d+)', cpd)

        for elem in match_mol:
            cpd_elem_list.append(elem[0])
            elem_weight_list.append(int(elem[1]))
        
        elem_mass_list = [mass_dict[x] for x in cpd_elem_list]
        cpd_mass = sum(np.multiply(elem_mass_list, elem_weight_list))
        # print(cpd_mass)
        ms1_mass_list.append(cpd_mass)
    
    rst_tab["ms1_mass"] = ms1_mass_list

    # calculate the basic ms2 mass for all the ms2 cpd
    ms2_mass_list = []
    for cpd in rst_tab["Fragment"]:
        cpd_elem_list = []
        elem_weight_list = []
        match_mol = re.findall(r'([A-Za-z]+)(\d+)', cpd)

        for elem in match_mol:
            cpd_elem_list.append(elem[0])
            elem_weight_list.append(int(elem[1]))

        elem_mass_list = [mass_dict[x] for x in cpd_elem_list]
        
        if args.polarity == "positive":
            cpd_mass = sum(np.multiply(elem_mass_list, elem_weight_list)) + mass_dict["H"]
        else:
            cpd_mass = sum(np.multiply(elem_mass_list, elem_weight_list)) - mass_dict["H"]
        # print(cpd_mass)
        ms2_mass_list.append(cpd_mass)

    rst_tab["ms2_mass"] = ms2_mass_list
    # print("rst_tab:\n", rst_tab)

    cpd_note_list = [] # list of tuple: (rst_tab["CompoundName"], rst_tab["BaseType"])
    note_cpd_list_dict = defaultdict(list) # dict of cpd list (rst_tab["BaseType"] : rst_tab["CompoundName"] list)
    ms1_mass_dict = defaultdict(float) # dict of ms1 list in float (cpd : ms1_mass)
    ms2_mass_dict = defaultdict(str) # dict of ms2 list string divided by space (cpd : ms2_mass string)

    for row in rst_tab.itertuples():
        cpd = row.CompoundName
        note = row.BaseType
        ms1_mass = row.ms1_mass
        ms2_mass = row.ms2_mass

        cpd_note_list.append((cpd, note))
        note_cpd_list_dict[note].append(cpd)
        ms1_mass_dict[cpd] = ms1_mass
        ms2_mass_dict[cpd] = str(ms2_mass)
    
    return cpd_note_list, note_cpd_list_dict, ms1_mass_dict, ms2_mass_dict


if __name__ == "__main__":
    class Arguments():
        def __init__(self, permethyl, nucleoside_type, polarity):
            if eval(permethyl.lower().capitalize()):
                self.analyte_form = "Permethyl"
            else:
                self.analyte_form = "Native"
            
            self.nucleoside_type = nucleoside_type

            self.polarity = polarity
            assert(self.polarity in ["positive", "negative"])

    
    args = Arguments("false", "DNA", "negative")
    print(args.analyte_form, args.nucleoside_type)

    input_path = "./data"
    cpd_note_list, note_cpd_list_dict, ms1_mass_dict, ms2_mass_dict = extrac_dataset_info(args, input_path, fn_cm='CustomizedMods.csv')

    print("cpd_note_list:", cpd_note_list)
    print("note_cpd_list_dict", note_cpd_list_dict)
    print("ms1_mass_dict", ms1_mass_dict)
    print("ms2_mass_dict", ms2_mass_dict)

    print("***** testing Unknown Search Mod *****")
    from NuMoFinder import unknown_search
    from NuMoFinder import Arguments_unknowMod
    from matchms.importing import load_from_mzxml
    from matchms import set_matchms_logger_level

    set_matchms_logger_level("ERROR")
    output_fd = "./test"
    polarity = "positive"
    min_rt = 10
    max_rt = 11
    min_mass = 200
    max_mass = 500
    min_height_unknow_search = 100
    permethyl = "false"
    nucleoside_type = "DNA"
    ppm_ms1 = 10
    ppm_ms2 = 10
    args_unknowMod = Arguments_unknowMod(output_fd, polarity, min_rt, max_rt, min_mass, max_mass, min_height_unknow_search, permethyl, nucleoside_type, ppm_ms1, ppm_ms2)

    input_fn = "./data/JL_S462_DNA_nucleoside_1/JL_S462_DNA_nucleoside_1.mzXML"
    spectrums_ms1_all_list = list(load_from_mzxml(input_fn, ms_level=1))    
    spectrums_ms2_all_list = list(load_from_mzxml(input_fn, ms_level=2))

    print("start unknow search mod...")
    unknown_search(args_unknowMod, spectrums_ms1_all_list, spectrums_ms2_all_list, ms2_mass_dict)



    




