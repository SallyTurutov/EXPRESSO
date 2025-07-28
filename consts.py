from enum import Enum

TISSUE_DATA_DIR = 'tissue_data'
GENE_DATA_DIR = 'gene_data'


class TISSUES(Enum):
    ADIPOSE_SUBCUTANEOUS = 'adipose_subcutaneous'
    ADIPOSE_VISCERAL_OMENTUM = 'adipose_visceral_omentum'
    ADRENAL_GLAND = 'adrenal_gland'
    ARTERY_AORTA = 'artery_aorta'
    ARTERY_CORONARY = 'artery_coronary'
    ARTERY_TIBIAL = 'artery_tibial'
    BLADDER = 'bladder'
    BRAIN_AMYGDALA = 'brain_amygdala'
    BRAIN_ANTERIOR_CINGULATE_CORTEX_BA24 = 'brain_anterior_cingulate_cortex_ba24'
    BRAIN_CAUDATE_BASAL_GANGLIA = 'brain_caudate_basal_ganglia'
    BRAIN_CEREBELLAR_HEMISPHERE = 'brain_cerebellar_hemisphere'
    BRAIN_CEREBELLUM = 'brain_cerebellum'
    BRAIN_CORTEX = 'brain_cortex'
    BRAIN_FRONTAL_CORTEX_BA9 = 'brain_frontal_cortex_ba9'
    BRAIN_HIPPOCAMPUS = 'brain_hippocampus'
    BRAIN_HYPOTHALAMUS = 'brain_hypothalamus'
    BRAIN_NUCLEUS_ACCUMBENS_BASAL_GANGLIA = 'brain_nucleus_accumbens_basal_ganglia'
    BRAIN_PUTAMEN_BASAL_GANGLIA = 'brain_putamen_basal_ganglia'
    BREAST_MAMMARY_TISSUE = 'breast_mammary_tissue'
    CELLS_CULTURED_FIBROBLASTS = 'cells_cultured_fibroblasts'
    CELLS_EBV_TRANSFORMED_LYMPHOCYTES = 'cells_ebv-transformed_lymphocytes'
    CERVIX_ECTOCERVIX = 'cervix_ectocervix'
    CERVIX_ENDOCERVIX = 'cervix_endocervix'
    COLON_SIGMOID = 'colon_sigmoid'
    COLON_TRANSVERSE = 'colon_transverse'
    ESOPHAGUS_GASTROESOPHAGEAL_JUNCTION = 'esophagus_gastroesophageal_junction'
    ESOPHAGUS_MUCOSA = 'esophagus_mucosa'
    ESOPHAGUS_MUSCULARIS = 'esophagus_muscularis'
    FALLICAN_TUBE = 'fallopian_tube'
    HEART_ATRIAL_APPENDAGE = 'heart_atrial_appendage'
    HEART_LEFT_VENTRICLE = 'heart_left_ventricle'
    KIDNEY_CORTEX = 'kidney_cortex'
    KIDNEY_MEDULLA = 'kidney_medulla'
    LIVER = 'liver'
    LUNG = 'lung'
    MINOR_SALIVARY_GLAND = 'minor_salivary_gland'
    MUSCLE_SKELETAL = 'muscle_skeletal'
    NERVE_TIBIAL = 'nerve_tibial'
    OVARY = 'ovary'
    PANCREAS = 'pancreas'
    PITUITARY = 'pituitary'
    PROSTATE = 'prostate'
    SKIN_NOT_SUN_EXPOSED_SUPRAPUBIC = 'skin_not_sun_exposed_suprapubic'
    SKIN_SUN_EXPOSED_LOWER_LEG = 'skin_sun_exposed_lower_leg'
    SMALL_INTESTINE_TERMINAL_ILEUM = 'small_intestine_terminal_ileum'
    SPLEEN = 'spleen'
    STOMACH = 'stomach'
    TESTIS = 'testis'
    THYROID = 'thyroid'
    UTERUS = 'uterus'
    VAGINA = 'vagina'
    WHOLE_BLOOD = 'whole_blood'


TISSUE_TO_DATA_DIR = {
    TISSUES.ADIPOSE_SUBCUTANEOUS: 'gene_tpm_v10_adipose_subcutaneous',
    TISSUES.ADIPOSE_VISCERAL_OMENTUM: 'gene_tpm_v10_adipose_visceral_omentum',
    TISSUES.ADRENAL_GLAND: 'gene_tpm_v10_adrenal_gland',
    TISSUES.ARTERY_AORTA: 'gene_tpm_v10_artery_aorta',
    TISSUES.ARTERY_CORONARY: 'gene_tpm_v10_artery_coronary',
    TISSUES.ARTERY_TIBIAL: 'gene_tpm_v10_artery_tibial',
    TISSUES.BLADDER: 'gene_tpm_v10_bladder',
    TISSUES.BRAIN_AMYGDALA: 'gene_tpm_v10_brain_amygdala',
    TISSUES.BRAIN_ANTERIOR_CINGULATE_CORTEX_BA24: 'gene_tpm_v10_brain_anterior_cingulate_cortex_ba24',
    TISSUES.BRAIN_CAUDATE_BASAL_GANGLIA: 'gene_tpm_v10_brain_caudate_basal_ganglia',
    TISSUES.BRAIN_CEREBELLAR_HEMISPHERE: 'gene_tpm_v10_brain_cerebellar_hemisphere',
    TISSUES.BRAIN_CEREBELLUM: 'gene_tpm_v10_brain_cerebellum',
    TISSUES.BRAIN_CORTEX: 'gene_tpm_v10_brain_cortex',
    TISSUES.BRAIN_FRONTAL_CORTEX_BA9: 'gene_tpm_v10_brain_frontal_cortex_ba9',
    TISSUES.BRAIN_HIPPOCAMPUS: 'gene_tpm_v10_brain_hippocampus',
    TISSUES.BRAIN_HYPOTHALAMUS: 'gene_tpm_v10_brain_hypothalamus',
    TISSUES.BRAIN_NUCLEUS_ACCUMBENS_BASAL_GANGLIA: 'gene_tpm_v10_brain_nucleus_accumbens_basal_ganglia',
    TISSUES.BRAIN_PUTAMEN_BASAL_GANGLIA: 'gene_tpm_v10_brain_putamen_basal_ganglia',
    TISSUES.BREAST_MAMMARY_TISSUE: 'gene_tpm_v10_breast_mammary_tissue',
    TISSUES.CELLS_CULTURED_FIBROBLASTS: 'gene_tpm_v10_cells_cultured_fibroblasts',
    TISSUES.CELLS_EBV_TRANSFORMED_LYMPHOCYTES: 'gene_tpm_v10_cells_ebv-transformed_lymphocytes',
    TISSUES.CERVIX_ECTOCERVIX: 'gene_tpm_v10_cervix_ectocervix',
    TISSUES.CERVIX_ENDOCERVIX: 'gene_tpm_v10_cervix_endocervix',
    TISSUES.COLON_SIGMOID: 'gene_tpm_v10_colon_sigmoid',
    TISSUES.COLON_TRANSVERSE: 'gene_tpm_v10_colon_transverse',
    TISSUES.ESOPHAGUS_GASTROESOPHAGEAL_JUNCTION: 'gene_tpm_v10_esophagus_gastroesophageal_junction',
    TISSUES.ESOPHAGUS_MUCOSA: 'gene_tpm_v10_esophagus_mucosa',
    TISSUES.ESOPHAGUS_MUSCULARIS: 'gene_tpm_v10_esophagus_muscularis',
    TISSUES.FALLICAN_TUBE: 'gene_tpm_v10_fallopian_tube',
    TISSUES.HEART_ATRIAL_APPENDAGE: 'gene_tpm_v10_heart_atrial_appendage',
    TISSUES.HEART_LEFT_VENTRICLE: 'gene_tpm_v10_heart_left_ventricle',
    TISSUES.KIDNEY_CORTEX: 'gene_tpm_v10_kidney_cortex',
    TISSUES.KIDNEY_MEDULLA: 'gene_tpm_v10_kidney_medulla',
    TISSUES.LIVER: 'gene_tpm_v10_liver',
    TISSUES.LUNG: 'gene_tpm_v10_lung',
    TISSUES.MINOR_SALIVARY_GLAND: 'gene_tpm_v10_minor_salivary_gland',
    TISSUES.MUSCLE_SKELETAL: 'gene_tpm_v10_muscle_skeletal',
    TISSUES.NERVE_TIBIAL: 'gene_tpm_v10_nerve_tibial',
    TISSUES.OVARY: 'gene_tpm_v10_ovary',
    TISSUES.PANCREAS: 'gene_tpm_v10_pancreas',
    TISSUES.PITUITARY: 'gene_tpm_v10_pituitary',
    TISSUES.PROSTATE: 'gene_tpm_v10_prostate',
    TISSUES.SKIN_NOT_SUN_EXPOSED_SUPRAPUBIC: 'gene_tpm_v10_skin_not_sun_exposed_suprapubic',
    TISSUES.SKIN_SUN_EXPOSED_LOWER_LEG: 'gene_tpm_v10_skin_sun_exposed_lower_leg',
    TISSUES.SMALL_INTESTINE_TERMINAL_ILEUM: 'gene_tpm_v10_small_intestine_terminal_ileum',
    TISSUES.SPLEEN: 'gene_tpm_v10_spleen',
    TISSUES.STOMACH: 'gene_tpm_v10_stomach',
    TISSUES.TESTIS: 'gene_tpm_v10_testis',
    TISSUES.THYROID: 'gene_tpm_v10_thyroid',
    TISSUES.UTERUS: 'gene_tpm_v10_uterus',
    TISSUES.VAGINA: 'gene_tpm_v10_vagina',
    TISSUES.WHOLE_BLOOD: 'gene_tpm_v10_whole_blood',
}

# Map fine-grained enum tissue name to the corresponding coarse tissue label in SMILES data
TISSUE_ENUM_TO_SMILES_LABEL = {
    TISSUES.BRAIN_HIPPOCAMPUS: "brain",
    TISSUES.BREAST_MAMMARY_TISSUE: "breast",
    TISSUES.CERVIX_ENDOCERVIX: "cervix",
    TISSUES.KIDNEY_CORTEX: "kidney",
    TISSUES.LIVER: "liver",
    TISSUES.LUNG: "lung",
    TISSUES.OVARY: "ovary",
    TISSUES.PROSTATE: "prostate",
    TISSUES.SKIN_NOT_SUN_EXPOSED_SUPRAPUBIC: "skin",
    TISSUES.ARTERY_TIBIAL: "vascular",
}

# For pretraining, use all tissues above
TISSUES_PRETRAIN_ALL = list(TISSUES)

TISSUES_TO_USE = [
    TISSUES.BRAIN_HIPPOCAMPUS,
    TISSUES.BREAST_MAMMARY_TISSUE,
    TISSUES.CERVIX_ENDOCERVIX,
    TISSUES.KIDNEY_CORTEX,
    TISSUES.LIVER,
    TISSUES.LUNG,
    TISSUES.OVARY,
    TISSUES.PROSTATE,
    TISSUES.SKIN_NOT_SUN_EXPOSED_SUPRAPUBIC,
    TISSUES.ARTERY_TIBIAL,
]

GENE_ALL_FILE = 'gene_all.txt'
GENE_SELECT_FILE = 'gene_select.txt'

PATHWAY_GENE_W = 'pathway_gene_weight.npy'
PATHWAY_CROSSTALK_NETWORK = 'pathway_crosstalk_network_matrix.npy'

SMILES_DATA_FILE = 'human_smiles_data.csv'
